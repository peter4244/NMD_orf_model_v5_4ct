#!/usr/bin/env python
"""
model_v6.py — the scanning-selection model.

Implements section 6 of analysis_plans/RETRAIN_PLAN_2026-08-01.md.

A transcript carries an ordered list of candidate reading frames, 5'->3'. Each
gets two numbers from sequence:

  p_k   capture — the probability a scanning ribosome initiates here, read from
        the start-codon window alone, through the initiation head's own encoder
  d_k   decay — the probability this transcript is degraded GIVEN that
        translation begins here, read from both windows and the structural block

Selection is stick-breaking: a ribosome reaches candidate k only by passing every
earlier one, so

    P(select k) = p_k * prod over j < k of (1 - p_j)
    P(NMD)      = sum over k of P(select k) * d_k

The residual mass — the ribosome passing every candidate — contributes nothing.

WHAT THIS ARCHITECTURE IS FOR. The model exists to be read, so two properties
matter more than accuracy. The initiation head sees only the start-codon window,
so its output is interpretable AS initiation rather than as "which ORF is real".
And the length axis is reduced by BINNED maximum pooling rather than one global
maximum, so a pattern's position survives to the resolution of one bin — a single
global maximum records only that a pattern occurred somewhere, which is why the
previous model could not learn a distance rule.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowEncoder(nn.Module):
    """Two convolutions, then a maximum within each of B bins along the length.

    B = 1 reproduces the previous architecture's global maximum, which discards
    position entirely. B > 1 keeps it to within one bin. The length axis is split
    with tensor_split, so a length not divisible by B gives the first few bins one
    extra position rather than failing or silently truncating.
    """

    def __init__(self, in_channels=9, conv_channels=32, n_bins=8, out_dim=32,
                 k1=15, k2=7, mid_pool=4, permute_bins=False):
        super().__init__()
        self.n_bins = n_bins
        self.permute_bins = permute_bins
        self.conv1 = nn.Conv1d(in_channels, conv_channels, k1, padding=k1 // 2)
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, k2, padding=k2 // 2)
        self.bn2 = nn.BatchNorm1d(conv_channels)
        self.mid_pool = nn.MaxPool1d(mid_pool) if mid_pool > 1 else nn.Identity()
        self.fc = nn.Linear(conv_channels * n_bins, out_dim)

    def forward(self, x, bin_perm=None):
        """bin_perm: (N, B) long, or (B,) broadcast over N. Supplied only by the
        mutagenesis bank, which needs ONE permutation held fixed across the
        unperturbed pass and every substitution of a transcript — an unpaired
        difference of two passes of this arm is dominated by the permutation
        rather than by the substitution. Supplying none redraws, which is what
        training does and what makes the arm a control.
        """
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.mid_pool(h)
        h = F.relu(self.bn2(self.conv2(h)))
        parts = torch.tensor_split(h, self.n_bins, dim=-1)
        pooled = torch.stack([p.amax(dim=-1) for p in parts], dim=1)  # (N, B, C)
        if bin_perm is not None and self.n_bins > 1:
            idx = bin_perm if bin_perm.dim() == 2 else bin_perm.expand(pooled.shape[0], -1)
            pooled = torch.gather(pooled, 1, idx.unsqueeze(-1).expand_as(pooled))
        elif self.permute_bins and self.n_bins > 1:
            # THE CONTROL ARM, and the permutation must be drawn PER EXAMPLE at
            # every forward pass. A permutation fixed at initialisation is not a
            # control: self.fc is fully connected over the flattened bins, so it
            # can undo any fixed reordering by permuting its own weights, and the
            # arm would measure nothing. Redrawing each time destroys the
            # correspondence between bin index and input feature while leaving
            # the parameter count and the value distribution identical.
            n, b, _ = pooled.shape
            idx = torch.argsort(torch.rand(n, b, device=pooled.device), dim=1)
            pooled = torch.gather(pooled, 1, idx.unsqueeze(-1).expand_as(pooled))
        return self.fc(pooled.flatten(1))


class ScanningNMDModel(nn.Module):
    """Per-candidate capture and decay, aggregated by stick-breaking.

    Args:
        n_structural: 1 for the interpretable variant, 5 for the predictor.
    """

    def __init__(self, n_seq_channels=9, conv_channels=32, n_bins=8,
                 seq_embed_dim=32, n_structural=1, decay_hidden=64,
                 dropout=0.2, permute_bins=False):
        super().__init__()
        enc = dict(in_channels=n_seq_channels, conv_channels=conv_channels,
                   n_bins=n_bins, out_dim=seq_embed_dim, permute_bins=permute_bins)

        # THREE ENCODERS, NOT TWO. The initiation head reads the ATG window
        # through its own weights rather than sharing the decay head's ATG
        # branch. Sharing would make "what the initiation head learned" and "what
        # the decay head learned" the same object, and the initiation head's
        # filters are what gets read.
        self.enc_init = WindowEncoder(**enc)
        self.enc_atg = WindowEncoder(**enc)
        self.enc_stop = WindowEncoder(**enc)

        self.init_head = nn.Linear(seq_embed_dim, 1)

        self.struct_fc = nn.Linear(n_structural, seq_embed_dim)
        self.decay_body = nn.Sequential(
            nn.Linear(seq_embed_dim * 3, decay_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.decay_head = nn.Linear(decay_hidden, 1)

    def forward(self, atg, stop, structural, mask, return_parts=False):
        """
        atg, stop:  (batch, K, 9, W)   candidates ordered 5'->3'
        structural: (batch, K, n_structural)
        mask:       (batch, K) bool, True where the slot holds a real candidate

        Returns the transcript logit, (batch,).
        """
        bsz, K = mask.shape
        flat = mask.reshape(-1)
        n_valid = int(flat.sum())

        z_p = atg.new_zeros(bsz * K)
        z_d = atg.new_zeros(bsz * K)
        if n_valid:
            a = atg.reshape(bsz * K, *atg.shape[2:])[flat]
            s = stop.reshape(bsz * K, *stop.shape[2:])[flat]
            u = structural.reshape(bsz * K, -1)[flat]

            # capture, from the start-codon window only
            z_p[flat] = self.init_head(self.enc_init(a)).squeeze(-1)

            # decay, from both windows and the structural block
            fused = torch.cat([self.enc_atg(a), self.enc_stop(s),
                               F.relu(self.struct_fc(u))], dim=-1)
            z_d[flat] = self.decay_head(self.decay_body(fused)).squeeze(-1)

        z_p = z_p.view(bsz, K)
        z_d = z_d.view(bsz, K)
        return self.aggregate(z_p, z_d, mask, return_parts=return_parts)

    def aggregate(self, z_p, z_d, mask, return_parts=False):
        """Stick-breaking over the candidates of each transcript, from the two
        per-candidate logits. Separated from forward() because the mutagenesis
        bank recomputes only the encoders of the candidates a substitution
        touches and then re-aggregates over the whole transcript — the product
        couples the candidates, so the aggregation cannot be cached even though
        the encoders can. One definition, called from both places.

        z_p, z_d: (batch, K)   mask: (batch, K) bool
        """
        # ---- stick-breaking, in log space --------------------------------
        # logsigmoid rather than log(sigmoid(.)): p_k saturates toward 0 and 1
        # and the naive form loses the log of a small number entirely.
        log_p = F.logsigmoid(z_p)                      # log p_k
        log_q = F.logsigmoid(-z_p)                     # log (1 - p_k)
        # A padded slot passes with probability 1, so it contributes 0 to the
        # running product and cannot shift the mass of the real candidates.
        log_q = log_q.masked_fill(~mask, 0.0)
        log_pass = torch.cumsum(log_q, dim=1) - log_q  # exclusive: sum over j < k

        log_sel = log_pass + log_p                     # log P(select k)
        log_sel = log_sel.masked_fill(~mask, float("-inf"))

        log_d = F.logsigmoid(z_d)
        log_terms = log_sel + log_d                    # log[ P(select k) d_k ]
        log_nmd = torch.logsumexp(log_terms, dim=1)    # log P(NMD)

        p_nmd = log_nmd.exp().clamp(1e-6, 1 - 1e-6)
        logit = torch.log(p_nmd) - torch.log1p(-p_nmd)

        if return_parts:
            return logit, dict(p=torch.sigmoid(z_p) * mask,
                               d=torch.sigmoid(z_d) * mask,
                               p_select=log_sel.exp(),
                               p_nmd=p_nmd,
                               residual=log_q.sum(dim=1).exp())
        return logit


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(**kw):
    m = ScanningNMDModel(**kw)
    print(f"ScanningNMDModel: {count_parameters(m):,} trainable parameters "
          f"(conv_channels={kw.get('conv_channels', 32)}, "
          f"n_bins={kw.get('n_bins', 8)}, "
          f"n_structural={kw.get('n_structural', 1)}, "
          f"permute_bins={kw.get('permute_bins', False)})")
    return m


if __name__ == "__main__":
    torch.manual_seed(0)
    W, K, B = 1000, 6, 3

    for n_bins in (1, 8, 16, 32):
        build_model(n_bins=n_bins)

    m = build_model(n_bins=8, n_structural=1)
    atg = torch.randn(B, K, 9, W)
    stop = torch.randn(B, K, 9, W)
    u = torch.randn(B, K, 1)
    mask = torch.ones(B, K, dtype=torch.bool)
    mask[0, 4:] = False          # 4 candidates
    mask[1, 1:] = False          # 1 candidate

    m.eval()
    with torch.no_grad():
        logit, parts = m(atg, stop, u, mask, return_parts=True)
    print(f"\n  logit {tuple(logit.shape)}   p_nmd {parts['p_nmd'].tolist()}")

    sel = parts["p_select"]
    tot = sel.sum(1)
    res = parts["residual"]
    print(f"  sum P(select) per transcript : {[round(v, 6) for v in tot.tolist()]}")
    print(f"  residual (passes everything) : {[round(v, 6) for v in res.tolist()]}")
    print(f"  sum + residual == 1          : "
          f"{torch.allclose(tot + res, torch.ones(B), atol=1e-5)}")
    print(f"  masked slots carry zero mass : "
          f"{bool((sel[~mask] == 0).all())}")

    # Does capture depend on the stop window? It must not: that restriction is
    # what licenses reading p_k as initiation rather than as "which ORF is real".
    # Tested by changing the stop window and nothing else.
    with torch.no_grad():
        _, r1 = m(atg, stop, u, mask, return_parts=True)
        _, r2 = m(atg, torch.randn_like(stop), u, mask, return_parts=True)
    print(f"  capture invariant to the stop window : "
          f"{torch.equal(r1['p'], r2['p'])}")
    print(f"  decay DOES move with it              : "
          f"{not torch.equal(r1['d'], r2['d'])}")

    # And the permuted-bin control has to actually change the output, or the arm
    # measures nothing. A permutation fixed at initialisation would not.
    mp = ScanningNMDModel(n_bins=8, n_structural=1, permute_bins=True)
    mp.load_state_dict(m.state_dict())
    mp.eval()
    with torch.no_grad():
        o1 = mp(atg, stop, u, mask)
        o2 = mp(atg, stop, u, mask)
    print(f"  permuted-bin arm redraws per pass    : {not torch.equal(o1, o2)}")

    m.train()
    loss = F.binary_cross_entropy_with_logits(
        m(atg, stop, u, mask), torch.tensor([1.0, 0.0, 1.0]))
    loss.backward()
    g_init = sum(q.grad.abs().sum().item() for q in m.enc_init.parameters()
                 if q.grad is not None)
    g_stop = sum(q.grad.abs().sum().item() for q in m.enc_stop.parameters()
                 if q.grad is not None)
    print(f"\n  backward: OK, loss {loss.item():.4f}")
    print(f"  gradient reaches enc_init {g_init:.3f} and enc_stop {g_stop:.3f}")

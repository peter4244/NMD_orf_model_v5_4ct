#!/usr/bin/env python3
"""
02_model.py — ORF-centric hybrid model for NMD prediction.

Architecture:
  Per-ORF encoder (shared weights):
    - ATG context CNN branch (sequence around start codon)
    - Stop context CNN branch (sequence around stop codon)
    - Structural feature branch (ORF-level features)
    → fused per-ORF embedding

  Attention aggregator:
    - Learns which ORFs matter (≈ ribosome selection probability)
    → transcript-level embedding

  Classification head:
    - Maps transcript embedding directly to NMD prediction
    → classification (NMD yes/no)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceCNN(nn.Module):
    """
    Small CNN for encoding a sequence context window.
    Input: (batch, n_channels, window_size)
    Output: (batch, out_dim)

    Kernel sizes adapt to window size:
      w20:  k=5, k=3
      w100: k=7, k=5
      w500: k=15, k=7 + extra pooling layer
    """

    def __init__(self, in_channels=9, conv_channels=32, out_dim=32, window_size=100):
        super().__init__()

        if window_size <= 20:
            k1, k2 = 5, 3
        elif window_size <= 100:
            k1, k2 = 7, 5
        else:
            k1, k2 = 15, 7

        self.conv1 = nn.Conv1d(in_channels, conv_channels, kernel_size=k1,
                               padding=k1 // 2)
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=k2,
                               padding=k2 // 2)
        self.bn2 = nn.BatchNorm1d(conv_channels)

        # For large windows, add a max-pool between conv layers to reduce length
        self.mid_pool = nn.MaxPool1d(4) if window_size > 100 else nn.Identity()

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(conv_channels, out_dim)

    def forward(self, x):
        """x: (batch, in_channels, window_size)"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.mid_pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # (batch, conv_channels)
        return self.fc(x)  # (batch, out_dim)


class ORFEncoder(nn.Module):
    """
    Shared-weight encoder for a single ORF.
    Takes ATG window, stop window, and structural features → ORF embedding.
    """

    def __init__(self, n_seq_channels=9, conv_channels=32, seq_embed_dim=32,
                 n_orf_features=4, orf_embed_dim=64, dropout=0.2,
                 window_size_atg=100, window_size_stop=100):
        super().__init__()

        self.atg_cnn = SequenceCNN(n_seq_channels, conv_channels, seq_embed_dim,
                                   window_size_atg)
        self.stop_cnn = SequenceCNN(n_seq_channels, conv_channels, seq_embed_dim,
                                    window_size_stop)
        self.struct_fc = nn.Linear(n_orf_features, seq_embed_dim)

        # Fusion: concat ATG + stop + structural embeddings
        fusion_dim = seq_embed_dim * 3
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, orf_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, atg_window, stop_window, orf_features):
        """
        Args:
            atg_window:   (batch, n_channels, window_size)
            stop_window:  (batch, n_channels, window_size)
            orf_features: (batch, n_orf_features)
        Returns:
            orf_embedding: (batch, orf_embed_dim)
        """
        atg_emb = self.atg_cnn(atg_window)
        stop_emb = self.stop_cnn(stop_window)
        struct_emb = F.relu(self.struct_fc(orf_features))

        fused = torch.cat([atg_emb, stop_emb, struct_emb], dim=-1)
        return self.fusion(fused)


class AttentionAggregator(nn.Module):
    """
    Attention-weighted pooling over ORF embeddings.
    Learns scalar attention scores per ORF, masked softmax, weighted sum.
    Attention weights are interpretable as ribosome selection probabilities.
    """

    def __init__(self, embed_dim=64):
        super().__init__()
        self.attn_score = nn.Linear(embed_dim, 1)

    def forward(self, orf_embeddings, orf_mask):
        """
        Args:
            orf_embeddings: (batch, K, embed_dim)
            orf_mask:       (batch, K) — True where ORF exists
        Returns:
            transcript_emb: (batch, embed_dim)
            attn_weights:   (batch, K) — normalized attention weights
        """
        scores = self.attn_score(orf_embeddings).squeeze(-1)  # (batch, K)

        # Masked softmax: set scores of padded ORFs to -inf
        scores = scores.masked_fill(~orf_mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)

        # Handle transcripts with zero ORFs (all -inf → NaN after softmax)
        attn_weights = attn_weights.nan_to_num(nan=0.0)

        # Weighted sum
        transcript_emb = torch.bmm(
            attn_weights.unsqueeze(1),   # (batch, 1, K)
            orf_embeddings               # (batch, K, embed_dim)
        ).squeeze(1)                     # (batch, embed_dim)

        return transcript_emb, attn_weights


class NMDOrfModel(nn.Module):
    """
    Full ORF-centric hybrid model for NMD prediction.

    Processes K ORFs per transcript through a shared encoder,
    aggregates via attention, and predicts NMD status.
    """

    def __init__(self, config):
        super().__init__()

        # Extract config
        n_seq_channels = config.get("n_seq_channels", 9)
        conv_channels = config.get("conv_channels", 32)
        seq_embed_dim = config.get("seq_embed_dim", 32)
        n_orf_features = config.get("n_orf_features", 4)
        orf_embed_dim = config.get("orf_embed_dim", 64)
        head_hidden = config.get("head_hidden", 32)
        dropout_orf = config.get("dropout_orf", 0.2)
        dropout_head = config.get("dropout_head", 0.3)
        window_size_atg = config.get("window_size_atg", config.get("window_size", 100))
        window_size_stop = config.get("window_size_stop", config.get("window_size", 100))

        # Shared ORF encoder
        self.orf_encoder = ORFEncoder(
            n_seq_channels=n_seq_channels,
            conv_channels=conv_channels,
            seq_embed_dim=seq_embed_dim,
            n_orf_features=n_orf_features,
            orf_embed_dim=orf_embed_dim,
            dropout=dropout_orf,
            window_size_atg=window_size_atg,
            window_size_stop=window_size_stop,
        )

        # Attention aggregator
        self.aggregator = AttentionAggregator(orf_embed_dim)

        # v5: classification head directly from transcript embedding (no TX bypass)
        self.head = nn.Sequential(
            nn.Linear(orf_embed_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_head),
        )
        self.cls_head = nn.Linear(head_hidden, 1)

    def forward(self, atg_windows, stop_windows, orf_features, orf_mask,
                return_attention=False):
        """
        v5: no tx_features input.

        Args:
            atg_windows:  (batch, K, n_channels, window_size)
            stop_windows: (batch, K, n_channels, window_size)
            orf_features: (batch, K, n_orf_features)
            orf_mask:     (batch, K) bool
            return_attention: if True, also return attention weights

        Returns:
            cls_logits: (batch, 1) — NMD classification logits
            attn_weights: (batch, K) — only if return_attention=True
        """
        batch_size, K = orf_mask.shape
        embed_dim = self.aggregator.attn_score.in_features

        flat_mask = orf_mask.reshape(-1)
        n_valid = flat_mask.sum().item()

        orf_emb = torch.zeros(batch_size * K, embed_dim,
                              device=atg_windows.device, dtype=atg_windows.dtype)

        if n_valid > 0:
            atg_flat = atg_windows.reshape(batch_size * K, *atg_windows.shape[2:])
            stop_flat = stop_windows.reshape(batch_size * K, *stop_windows.shape[2:])
            feat_flat = orf_features.reshape(batch_size * K, -1)

            valid_atg = atg_flat[flat_mask]
            valid_stop = stop_flat[flat_mask]
            valid_feat = feat_flat[flat_mask]

            valid_emb = self.orf_encoder(valid_atg, valid_stop, valid_feat)
            orf_emb[flat_mask] = valid_emb.to(orf_emb.dtype)

        orf_emb = orf_emb.reshape(batch_size, K, embed_dim)

        # Attention-weighted aggregation
        transcript_emb, attn_weights = self.aggregator(orf_emb, orf_mask)

        # Classification (no TX bypass)
        hidden = self.head(transcript_emb)
        cls_logits = self.cls_head(hidden)

        if return_attention:
            return cls_logits, attn_weights
        return cls_logits


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(config):
    """Build model from config dict and print summary."""
    model = NMDOrfModel(config)
    n_params = count_parameters(model)
    print(f"NMDOrfModel: {n_params:,} trainable parameters")
    ws_atg = config.get("window_size_atg", config.get("window_size", 100))
    ws_stop = config.get("window_size_stop", config.get("window_size", 100))
    print(f"  Window sizes: ATG={ws_atg}, Stop={ws_stop}")
    print(f"  ORF embed dim: {config.get('orf_embed_dim', 64)}")
    print(f"  Conv channels: {config.get('conv_channels', 32)}")
    return model


if __name__ == "__main__":
    # Quick smoke test with dummy data
    config = {
        "n_seq_channels": 9,
        "conv_channels": 32,
        "seq_embed_dim": 32,
        "n_orf_features": 4,
        "orf_embed_dim": 64,
        "head_hidden": 32,
        "dropout_orf": 0.2,
        "dropout_head": 0.3,
        "window_size_atg": 100,
        "window_size_stop": 1000,
    }

    model = build_model(config)

    # Dummy forward pass
    batch, K, C = 4, 5, 9
    W_atg, W_stop = 100, 1000
    atg = torch.randn(batch, K, C, W_atg)
    stop = torch.randn(batch, K, C, W_stop)
    orf_feat = torch.randn(batch, K, 4)
    mask = torch.ones(batch, K, dtype=torch.bool)
    mask[0, 3:] = False  # simulate transcript with only 3 ORFs
    mask[1, :] = False    # simulate transcript with 0 ORFs

    cls_logits, attn = model(atg, stop, orf_feat, mask, return_attention=True)

    print(f"\n  cls_logits: {cls_logits.shape}")
    print(f"  attn:       {attn.shape}")
    print(f"  attn[0]:    {attn[0].detach().numpy().round(3)}")
    print(f"  attn[1]:    {attn[1].detach().numpy().round(3)} (0-ORF transcript)")

    # Test backward pass
    model.train()
    cls_out = model(atg, stop, orf_feat, mask)
    loss = cls_out.sum()
    loss.backward()
    print("  backward pass: OK")

    # Test batch_size=1 (BN edge case)
    model.train()
    cls_out = model(atg[:1], stop[:1], orf_feat[:1], mask[:1])
    print(f"  batch=1: OK, cls={cls_out.item():.4f}")

    # Test eval mode
    model.eval()
    with torch.no_grad():
        cls_out = model(atg, stop, orf_feat, mask)
    print(f"  eval mode: OK")

    # Test asymmetric window sizes
    for ws_atg, ws_stop in [(100, 500), (500, 1000), (100, 2000)]:
        cfg = {**config, "window_size_atg": ws_atg, "window_size_stop": ws_stop}
        m = build_model(cfg)
        out = m(torch.randn(2, K, C, ws_atg), torch.randn(2, K, C, ws_stop),
                orf_feat[:2], mask[:2])
        print(f"  atg={ws_atg}, stop={ws_stop}: OK, params={count_parameters(m):,}")

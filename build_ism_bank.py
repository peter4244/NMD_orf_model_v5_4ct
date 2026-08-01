#!/usr/bin/env python
"""
build_ism_bank.py — the in-silico mutagenesis bank.

Implements section 9 of analysis_plans/RETRAIN_PLAN_2026-08-01.md. One invocation
builds one arm. The interpretation window computes every metric from the output;
nothing here interprets anything.

WHAT ONE ENTRY MEANS. vals[i, p, b] is the change in transcript i's logit when the
base at 1-based transcript position p+1 is replaced by ACGT[b], IN EVERY CANDIDATE
WINDOW THAT CONTAINS THAT COORDINATE. Perturbing one window and not the others
would present the same coordinate as two different bases inside one forward pass,
which is a state no transcript can occupy, and it is the off-manifold condition
this method exists to avoid.

WHAT IS HELD FIXED. The junction channel, the reading-frame channels, the
structural block, the candidate coordinates and the candidate set itself. A
substitution can create or destroy an ATG or a stop codon; the pool is not
re-derived when it does. The bank measures the model's response to its input, not
to a re-scanned transcript.

THE CACHE, AND WHY THE NO-OP FLOOR CHECKS IT. In eval mode a candidate's
embeddings depend on nothing but its own windows, so a candidate no window of
which contains p keeps the values from the unperturbed pass. That is an identity,
not an approximation — but the cached path runs the encoder at a different batch
size from the base pass, and a different batch size can pick a different reduction
order. Substituting the observed base for itself exercises exactly that path and
must return zero, so the floor measures the error the cache introduces rather than
merely checking an index.

Usage:
    python build_ism_bank.py --tensor results_tensor_chr21 \\
        --checkpoint runs/interp_c32_b8_s100/best.pt \\
        --split results_ism_v6/discovery_confirmation_split.tsv \\
        --n 1000 --out results_ism_v6/bank_interpretable.h5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from model_v6 import ScanningNMDModel
from tensor_io import decode_windows

ATG_LEFT, STOP_LEFT = 900, 500
WINDOW = 1000
NT = "ACGT"
STRUCTURAL_COLS = ["n_downstream_ejc", "is_ref_cds", "is_sqanti_cds",
                   "frac_start", "frac_stop"]
INTERPRETABLE, PREDICTOR = [0], [0, 1, 2, 3, 4]
CLAMP_LOGIT = 13.815511057963775      # log((1-1e-6)/1e-6), aggregate()'s clamp


# ------------------------------------------------------------------ geometry
def window_spans(orf_start, orf_end, tx_len):
    """The transcript positions each window actually holds a base for.

    Read off build_tensor.py's two encode_window_codes calls: the ATG window is
    anchored at orf_start with 900 to its left and filled no further than the ORF
    midpoint; the stop window is anchored at orf_end-1 with 500 to its left and
    filled no earlier than one past the midpoint. `mid` belongs to the ATG window,
    so the two never hold the same coordinate.
    """
    mid = (orf_start + orf_end) // 2
    a_lo = np.maximum(1, orf_start - ATG_LEFT)
    a_hi = np.minimum(np.minimum(tx_len, mid), orf_start + (WINDOW - ATG_LEFT) - 1)
    s_lo = np.maximum(mid + 1, (orf_end - 1) - STOP_LEFT)
    s_hi = np.minimum(tx_len, (orf_end - 1) + (WINDOW - STOP_LEFT) - 1)
    return a_lo, a_hi, s_lo, s_hi


def covering_index(a_lo, a_hi, s_lo, s_hi):
    """For every transcript position, which (candidate, window) pairs hold it.

    Returned as a sorted flat list plus per-position offsets, so the pairs for a
    block of positions are one contiguous slice.
    """
    pos, cand, win = [], [], []
    for k in range(len(a_lo)):
        if a_hi[k] >= a_lo[k]:
            r = np.arange(a_lo[k], a_hi[k] + 1)
            pos.append(r); cand.append(np.full(len(r), k)); win.append(np.zeros(len(r), np.int8))
        if s_hi[k] >= s_lo[k]:
            r = np.arange(s_lo[k], s_hi[k] + 1)
            pos.append(r); cand.append(np.full(len(r), k)); win.append(np.ones(len(r), np.int8))
    if not pos:
        return (np.zeros(0, np.int64),) * 3
    pos = np.concatenate(pos); cand = np.concatenate(cand); win = np.concatenate(win)
    o = np.lexsort((cand, pos))
    pos, cand, win = pos[o], cand[o], win[o]
    # A (position, candidate) pair must occur at most once. The midpoint rule
    # makes a candidate's two windows disjoint, so it holds by construction — but
    # the assembly of z_d writes z_d[perturbation, candidate] by index, and a
    # repeat would silently keep whichever was written last rather than fail.
    dup = (pos[1:] == pos[:-1]) & (cand[1:] == cand[:-1])
    assert not dup.any(), (
        f"{int(dup.sum())} (position, candidate) pairs are covered twice: a "
        f"candidate's ATG and stop windows overlap, which the midpoint clip of "
        f"plan §5.3 step 1 forbids")
    return pos, cand, win


# ------------------------------------------------------------------ the model
def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    cols = INTERPRETABLE if a["variant"] == "interpretable" else PREDICTOR
    if a.get("blank_junctions"):
        cols = []
    m = ScanningNMDModel(conv_channels=a["conv_channels"], n_bins=a["n_bins"],
                         n_structural=max(len(cols), 1),
                         permute_bins=bool(a.get("permute_bins", False)))
    m.load_state_dict(ck["model"]); m.to(device).eval()
    return m, ck, (cols if cols else [0]), a


class Encoders:
    """The three encoder calls, with an optional fixed bin permutation.

    The permuted-bin arm redraws its permutation at every forward pass, so one
    pass of it is a draw and not a prediction. The bank holds one draw fixed
    across the unperturbed pass and every substitution of a transcript and
    averages over draws; that pairing is what leaves the substitution as the only
    term that moves.
    """

    def __init__(self, model, perms=None):
        self.m = model
        self.perms = perms or {}

    def _p(self, name, idx):
        q = self.perms.get(name)
        return None if q is None else q[idx]

    def init(self, x, idx):
        return self.m.enc_init(x, bin_perm=self._p("init", idx))

    def atg(self, x, idx):
        return self.m.enc_atg(x, bin_perm=self._p("atg", idx))

    def stop(self, x, idx):
        return self.m.enc_stop(x, bin_perm=self._p("stop", idx))


def draw_perms(model, K, generator, device):
    """One permutation per candidate per encoder, for the control arm."""
    if not model.enc_init.permute_bins or model.enc_init.n_bins <= 1:
        return None
    B = model.enc_init.n_bins
    return {n: torch.argsort(torch.rand(K, B, generator=generator, device=device), dim=1)
            for n in ("init", "atg", "stop")}


# ------------------------------------------------------------------ one arm
def transcript_bank(model, enc, codes, orf_start, orf_end, tx_len, struct,
                    device, chunk_rows, rng):
    """The bank for one transcript. Returns vals, valid, obs, base_logit, floor."""
    # THE CACHE IS EXACT ONLY IN EVAL MODE. In train mode batch normalization
    # uses batch statistics, so a candidate's embedding depends on which other
    # candidates share its batch -- and the cached path recomputes touched
    # candidates in a DIFFERENT batch composition from the base pass, so every
    # reused embedding would be subtly wrong and nothing would say so.
    assert not model.training, "model must be in eval mode; the cache assumes it"
    K = len(orf_start)
    a_lo, a_hi, s_lo, s_hi = window_spans(orf_start, orf_end, tx_len)
    pos, cand, win = covering_index(a_lo, a_hi, s_lo, s_hi)
    P = int(max(a_hi.max(), s_hi.max()))          # last covered position

    # ---- the unperturbed pass, and the embeddings the cache keeps -----------
    atg9 = torch.as_tensor(decode_windows(codes[:, 0], orf_start, ATG_LEFT, orf_start),
                           device=device)
    stop9 = torch.as_tensor(decode_windows(codes[:, 1], orf_end - 1, STOP_LEFT, orf_start),
                            device=device)
    kidx = torch.arange(K, device=device)
    u = torch.as_tensor(struct, dtype=torch.float32, device=device)
    with torch.no_grad():
        e_init = enc.init(atg9, kidx)
        e_atg = enc.atg(atg9, kidx)
        e_stop = enc.stop(stop9, kidx)
        u_emb = torch.relu(model.struct_fc(u))
        z_p0 = model.init_head(e_init).squeeze(-1)
        z_d0 = model.decay_head(model.decay_body(
            torch.cat([e_atg, e_stop, u_emb], dim=-1))).squeeze(-1)
        mask1 = torch.ones(1, K, dtype=torch.bool, device=device)
        base_logit = float(model.aggregate(z_p0[None].double(), z_d0[None].double(),
                                           mask1, stable=True).item())
        base_train = float(model.aggregate(z_p0[None], z_d0[None], mask1).item())

    # ---- observed base and validity, from the codes themselves -------------
    obs = np.full(P, -1, dtype=np.int8)
    valid = np.zeros(P, dtype=bool)
    fill = (codes & 7)
    for k in range(K):
        for w, lo, hi, anchor, left in ((0, a_lo[k], a_hi[k], orf_start[k], ATG_LEFT),
                                        (1, s_lo[k], s_hi[k], orf_end[k] - 1, STOP_LEFT)):
            if hi < lo:
                continue
            p = np.arange(lo, hi + 1)
            st = fill[k, w, p - anchor + left]
            # states are 0 unfilled, 1-4 ACGT, 5 filled but not ACGT (tensor_io).
            # State 5 must map to -1 and not to 4: a bare `st - 1` makes it index 4,
            # which is not a base and which passes an `obs >= 0` validity test.
            obs[p - 1] = np.where((st >= 1) & (st <= 4), st - 1, -1).astype(np.int8)
            valid[p - 1] = True
    # A position holding something other than ACGT has no observed base to
    # substitute for itself, so it carries no floor and is not measurable here.
    valid &= obs >= 0

    vals = np.full((P, 4), np.nan, dtype=np.float32)

    # ---- the substitutions --------------------------------------------------
    keep = valid[pos - 1]
    pos_v, cand_v, win_v = pos[keep], cand[keep], win[keep]
    order = np.argsort(pos_v, kind="stable")
    pos_v, cand_v, win_v = pos_v[order], cand_v[order], win_v[order]

    # EVERY POSITION CARRIES ITS OWN NO-OP, AND THE EFFECT IS MEASURED AGAINST IT.
    # The encoder's output for a fixed input row depends on how many rows share
    # its batch -- measured on this model, three regimes at batch 1, 2-7 and 8+,
    # 3.28e-07 apart. The unperturbed pass runs at batch K and a chunk runs at
    # batch len(chunk), so a difference taken between them carries that offset
    # whenever K and the chunk fall in different regimes. With K under 8 for
    # 15.5% of transcripts, that is a systematic error CORRELATED WITH CANDIDATE
    # COUNT, not noise.
    #
    # Substituting the observed base for itself in the SAME chunk gives a
    # baseline computed at the same batch shape, and the offset cancels exactly.
    # Row position within a batch does not matter -- verified -- so same-chunk is
    # enough and the rows need no particular order. This is why all four bases
    # are computed rather than three: the fourth is the baseline, not a check.
    noop_max = 0.0
    starts = np.searchsorted(pos_v, np.unique(pos_v))
    upos = np.unique(pos_v)

    i = 0
    while i < len(upos):
        # each position contributes 4 rows per covering pair: 3 substitutions and
        # the no-op that is their baseline
        j, rows = i, 0
        while j < len(upos):
            lo = starts[j]
            hi = starts[j + 1] if j + 1 < len(starts) else len(pos_v)
            r = (hi - lo) * 4
            if rows and rows + r > chunk_rows:
                break
            rows += r; j += 1
        block = upos[i:j]
        lo, hi = starts[i], (starts[j] if j < len(starts) else len(pos_v))
        bp, bc, bw = pos_v[lo:hi], cand_v[lo:hi], win_v[lo:hi]

        # expand each covering pair over the bases being substituted
        rc, rw, rb, rp = [], [], [], []
        for n, p in enumerate(block):
            m = bp == p
            o = int(obs[p - 1])
            for b in range(4):          # the observed base included, as the baseline
                c = int(m.sum())
                rc.append(bc[m]); rw.append(bw[m]); rb.append(np.full(c, b))
                rp.append((n, b, c))
        # perturbation ids, one per (position, base)
        pid = np.concatenate([np.full(c, t) for t, (_, _, c) in enumerate(rp)])
        rc = np.concatenate(rc); rw = np.concatenate(rw); rb = np.concatenate(rb)
        rpos = np.concatenate([np.full(c, block[n]) for (n, _, c) in rp])
        n_pert = len(rp)

        # build the perturbed codes: bits 0-2 take the new base, bit 3 (junction)
        # is a property of the annotation and is preserved
        anchor = np.where(rw == 0, orf_start[rc], orf_end[rc] - 1)
        left = np.where(rw == 0, ATG_LEFT, STOP_LEFT)
        widx = rpos - anchor + left
        pc = codes[rc, rw].copy()
        pc[np.arange(len(pc)), widx] = (pc[np.arange(len(pc)), widx] & 8) | (rb + 1)

        is_atg = rw == 0
        z_p = z_p0[None].repeat(n_pert, 1).clone()
        z_d = z_d0[None].repeat(n_pert, 1).clone()
        with torch.no_grad():
            if is_atg.any():
                sel = np.flatnonzero(is_atg)
                x = torch.as_tensor(decode_windows(pc[sel], orf_start[rc[sel]],
                                                   ATG_LEFT, orf_start[rc[sel]]),
                                    device=device)
                ci = torch.as_tensor(rc[sel], device=device, dtype=torch.long)
                pi = torch.as_tensor(pid[sel], device=device, dtype=torch.long)
                ei = enc.init(x, ci)
                ea = enc.atg(x, ci)
                z_p[pi, ci] = model.init_head(ei).squeeze(-1)
                z_d[pi, ci] = model.decay_head(model.decay_body(
                    torch.cat([ea, e_stop[ci], u_emb[ci]], dim=-1))).squeeze(-1)
            if (~is_atg).any():
                sel = np.flatnonzero(~is_atg)
                x = torch.as_tensor(decode_windows(pc[sel], orf_end[rc[sel]] - 1,
                                                   STOP_LEFT, orf_start[rc[sel]]),
                                    device=device)
                ci = torch.as_tensor(rc[sel], device=device, dtype=torch.long)
                pi = torch.as_tensor(pid[sel], device=device, dtype=torch.long)
                es = enc.stop(x, ci)
                z_d[pi, ci] = model.decay_head(model.decay_body(
                    torch.cat([e_atg[ci], es, u_emb[ci]], dim=-1))).squeeze(-1)
            mask = torch.ones(n_pert, K, dtype=torch.bool, device=device)
            out = model.aggregate(z_p.double(), z_d.double(), mask,
                                  stable=True).cpu().numpy()

        # THE DIFFERENCE IS TAKEN AGAINST THE NO-OP FROM THIS SAME CHUNK, never
        # against the batch-K base pass. row_of maps (position index, base) to the
        # row of `out` so the pairing is explicit rather than positional.
        row_of = {(n, b): t for t, (n, b, _) in enumerate(rp)}
        for n, p0 in enumerate(block):
            p = int(p0)
            o = int(obs[p - 1])
            ref = float(out[row_of[(n, o)]])
            # how far this chunk's baseline sits from the batch-K base pass. It is
            # the offset the same-chunk baseline removes, and it is reported so
            # its size is visible rather than assumed small.
            noop_max = max(noop_max, abs(ref - base_logit))
            for b in range(4):
                if b != o:
                    vals[p - 1, b] = float(out[row_of[(n, b)]]) - ref
        i = j

    return (vals, valid, obs, (base_logit, base_train), (noop_max, int(valid.sum())),
            (a_lo, a_hi, s_lo, s_hi))


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="results_ism_v6/discovery_confirmation_split.tsv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--chunk-rows", type=int, default=4096)
    ap.add_argument("--perm-draws", type=int, default=1,
                    help="control arm only: paired permutation draws to average over")
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--device", default="")
    ap.add_argument("--limit-transcripts", type=int, default=0)
    ap.add_argument("--only", default="",
                    help="comma-separated isoform_ids; for testing the geometry "
                         "extremes rather than the head of the order")
    ap.add_argument("--from-index", type=int, default=0, dest="from_index",
                    help="build only take[from:to); shards let tasks share a directory")
    ap.add_argument("--to-index", type=int, default=0, dest="to_index")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    model, ck, cols, ckargs = load_model(args.checkpoint, device)
    is_control = bool(ckargs.get("permute_bins", False))
    R = args.perm_draws if is_control else 1
    print(f"model      {args.checkpoint}")
    print(f"  variant {ckargs['variant']}  conv_channels {ckargs['conv_channels']}  "
          f"n_bins {ckargs['n_bins']}  permute_bins {is_control}  seed {ckargs['seed']}")
    print(f"  structural columns {[STRUCTURAL_COLS[c] for c in cols]}")
    if is_control:
        print(f"  CONTROL ARM: {R} paired permutation draw(s), averaged")
    if ckargs.get("blank_sequence"):
        raise SystemExit(
            "the sequence-blanked arm has no bank: it is trained and evaluated with "
            "channels 0-3 and 5 zeroed, and a substitution changes channels 0-3 and 5 "
            "and nothing else, so every entry would be exactly zero by construction "
            "(plan §9.3 step 8)")

    # ------------------------------------------------------------- the subset
    sp_all = pd.read_csv(args.split, sep="\t")
    sp = sp_all.sort_values("rank", kind="stable")
    with h5py.File(str(Path(args.tensor) / "nmd_tensor.h5"), "r") as f:
        iso = np.array([s.decode() for s in f["isoform_id"][:]])
        row = {s: i for i, s in enumerate(iso)}
        offset, count = f["offset"][:], f["count"][:]
        split_lab = np.array([s.decode() for s in f["split"][:]])
        gene = np.array([s.decode() for s in f["gene_id"][:]])
        labels = f["labels"][:]
        o_start_all, o_end_all = f["orf_start"][:], f["orf_end"][:]
        struct_all = f["structural"][:]
        codes_all = f["codes"][:]
        atg_left = int(f.attrs["atg_left"])
        pool_sha = f.attrs.get("pool_sha256", "")
    assert atg_left == ATG_LEFT, f"tensor anchors at {atg_left}, this script assumes {ATG_LEFT}"

    sp = sp[sp["isoform_id"].isin(row)]
    if args.only:
        want = [x for x in args.only.split(",") if x]
        take = sp[sp["isoform_id"].isin(want)]
        missing_only = sorted(set(want) - set(take["isoform_id"]))
        if missing_only:
            raise SystemExit(f"--only: not in the split/tensor: {missing_only}")
    else:
        take = sp.head(args.n if args.n > 0 else len(sp))
        if args.limit_transcripts:
            take = take.head(args.limit_transcripts)
    print(f"\nsubset     {len(take):,} transcripts "
          f"(rank < {int(take['rank'].max())+1:,} of the fixed order)")
    print(f"  in the tensor {len(sp):,} of {len(sp_all):,} split rows; "
          f"prevalence {take['is_nmd'].mean():.4f}")
    for a in ("discovery", "confirmation"):
        print(f"  {a:<13} {int((take['arm'] == a).sum()):>6,}")

    # --------------------------------------------------------------- the bank
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=device); gen.manual_seed(args.seed)

    chunk_rows = args.chunk_rows
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ONE SHARD PER TRANSCRIPT, and a restart skips what is already on disk.
    # The gpu partition kills at 8 hours and this run is longer than one
    # transcript, so a build with no resume is a build that starts from zero
    # every time the queue preempts it. The shard is written to a temporary name
    # and renamed, so a shard is never half-written. Splitting the subset across
    # array tasks with --from/--to writes into the same directory and needs no
    # coordination, because a shard is named by its transcript.
    shard_dir = Path(str(args.out) + ".shards")
    shard_dir.mkdir(parents=True, exist_ok=True)
    lo_i = args.from_index
    hi_i = args.to_index if args.to_index > 0 else len(take)
    todo = list(take.itertuples())[lo_i:hi_i]
    print(f"\nbuilding transcripts [{lo_i}, {hi_i}) of {len(take):,}")
    print(f"  shards in {shard_dir}")
    n_skipped = 0
    for n, r in enumerate(todo, start=lo_i):
        shard = shard_dir / f"{r.isoform_id}.npz"
        if shard.exists():
            n_skipped += 1
            continue
        i = row[r.isoform_id]
        sl = slice(int(offset[i]), int(offset[i]) + int(count[i]))
        cds, os_, oe_ = codes_all[sl], o_start_all[sl].astype(np.int64), o_end_all[sl].astype(np.int64)
        struct_k = struct_all[sl][:, cols]

        per_draw, base_draws = [], []
        for d in range(R):
            enc = Encoders(model, draw_perms(model, len(os_), gen, device) if is_control else None)
            # An OOM costs the chunk, not the run: the chunk is halved and retried,
            # the same backoff train_v6.py uses. Chunk size bounds memory and
            # nothing else, so shrinking it changes wall time and not a number.
            v = valid = None
            for attempt in range(5):
                try:
                    v, valid, obs, base, noop, spans = transcript_bank(
                        model, enc, cds, os_, oe_, int(r.tx_length), struct_k,
                        device, chunk_rows, rng)
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    chunk_rows = max(64, chunk_rows // 2)
                    print(f"      OOM on {r.isoform_id}, chunk_rows -> {chunk_rows}",
                          flush=True)
            if v is None:
                raise RuntimeError(f"{r.isoform_id}: OOM at chunk_rows={chunk_rows}")
            per_draw.append(v); base_draws.append(base[0])
        vals = per_draw[0] if R == 1 else np.nanmean(np.stack(per_draw), axis=0)
        spread = np.full(3, np.nan, np.float64)
        if R > 1:
            # Spread over the paired draws, on entries finite in EVERY draw.
            # nan-functions here return NaN for the all-NaN slices that invalid
            # positions and the observed base leave behind, and one NaN poisons
            # the mean -- which is how this reported "nan" the first time it ran.
            stk = np.stack(per_draw)
            ok = np.isfinite(stk).all(axis=0)
            if ok.any():
                spread = np.array([float(stk[:, ok].std(axis=0).mean()),
                                   float(np.abs(stk[:, ok].mean(axis=0)).mean()),
                                   float(np.std(base_draws))])

        a_lo, a_hi, s_lo, s_hi = spans
        # The temporary name must itself end in .npz: np.savez APPENDS ".npz" to
        # any path that does not, so a ".npz.tmp" target is written to
        # ".npz.tmp.npz" and the rename that follows it fails on a missing file.
        tmp = shard_dir / f".partial_{r.isoform_id}.npz"
        np.savez(tmp, vals=vals, valid=valid, obs=obs,
                 base_logit=np.float64(base[0]),
                 base_logit_training=np.float64(base[1]),
                 spans=np.stack([a_lo, a_hi, s_lo, s_hi], 1).astype(np.int32),
                 noop=np.float64(noop[0]), n_floor=np.int64(noop[1]),
                 spread=spread)
        os.replace(tmp, shard)            # a shard is never half-written
        if (n - lo_i) % 25 == 0 or n == hi_i - 1:
            print(f"  [{n+1:>5,}/{len(take):,}] {r.isoform_id:<32} K={len(os_):>3} "
                  f"P={len(valid):>6,} valid={int(valid.sum()):>6,} "
                  f"floor={noop[0]:.2e}  ({time.time()-t0:.0f}s)", flush=True)
    if n_skipped:
        print(f"  {n_skipped:,} transcripts already had a shard and were not recomputed")

    # ------------------------------------------------------------------ assemble
    missing = [r.isoform_id for r in take.itertuples()
               if not (shard_dir / f"{r.isoform_id}.npz").exists()]
    if missing:
        print(f"\n{len(missing):,} of {len(take):,} shards are missing — "
              f"not assembling. Rerun to fill them, or pass --from/--to for the gap.")
        print(f"  first missing: {missing[:3]}")
        return

    recs, spans_rows, floors, spreads = [], [], [], []
    n_floor_samples = 0
    for n, r in enumerate(take.itertuples()):
        z = np.load(shard_dir / f"{r.isoform_id}.npz")
        i = row[r.isoform_id]
        recs.append(dict(isoform_id=r.isoform_id, vals=z["vals"], valid=z["valid"],
                         obs=z["obs"], base_logit=float(z["base_logit"]),
                         base_train=float(z["base_logit_training"]),
                         label=int(labels[i]), arm=r.arm, split=split_lab[i],
                         gene=gene[i], K=len(z["spans"])))
        for k, (alo, ahi, slo, shi) in enumerate(z["spans"]):
            spans_rows.append((n, k, alo, ahi, slo, shi))
        floors.append(float(z["noop"])); n_floor_samples += int(z["n_floor"])
        if np.isfinite(z["spread"]).all():
            spreads.append(tuple(z["spread"].tolist()))

    W = max(len(x["valid"]) for x in recs)
    N = len(recs)
    vals = np.full((N, W, 4), np.nan, np.float32)
    valid = np.zeros((N, W), bool)
    obs = np.full((N, W), -1, np.int8)
    for i, x in enumerate(recs):
        p = len(x["valid"])
        vals[i, :p] = x["vals"]; valid[i, :p] = x["valid"]; obs[i, :p] = x["obs"]

    cand_count = np.array([x["K"] for x in recs], np.int32)
    cand_offset = np.concatenate([[0], np.cumsum(cand_count)])[:-1].astype(np.int32)
    floor = float(max(floors))

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(outp, "w") as f:
        f.create_dataset("vals", data=vals, compression="lzf")
        f.create_dataset("valid", data=valid, compression="lzf")
        f.create_dataset("obs", data=obs, compression="lzf")
        f.create_dataset("labels", data=np.array([x["label"] for x in recs], np.int8))
        f.create_dataset("base_logit",
                         data=np.array([x["base_logit"] for x in recs], np.float64))
        f.create_dataset("base_logit_training",
                         data=np.array([x["base_train"] for x in recs], np.float32))
        # WHICH TRANSCRIPTS THE TRAINING OUTPUT WOULD HAVE PINNED.
        #
        # Keyed on the STABLE logit, not the training one: the training logit can
        # never exceed 13.8155 by construction, so a threshold on it would catch
        # only the exactly-pinned and miss everything approaching the clamp.
        #
        # WHAT THIS FLAG DOES NOT COVER, measured on a z_d sweep at K=6. Near the
        # clamp the training path does not only pin, it COMPRESSES: at z_d = -16
        # with a step of 5.0 the training delta is 0.26x the true one, at -15 it
        # is 0.60x, at -14 it is 0.93x. Every one of those IS flagged here, so the
        # low end is covered. The high end is not: at z_d = +13 to +16 the stable
        # base logit is 4.14, nowhere near the clamp, the flag is correctly false,
        # and the training delta is still wrong by 1.12x to 1.44x and reaches
        # exactly 0.0 at +16. That is the float32 round-trip through P, not the
        # clamp, and no threshold on the base logit can find it.
        #
        # So the flag means "the clamp is active", not "the training path agrees
        # here". base_logit and base_logit_training are BOTH shipped so the
        # disagreement can be computed rather than inferred from the flag. vals
        # use the stable path throughout and carry none of this.
        pinned = np.abs(np.array([x["base_logit"] for x in recs])) >= CLAMP_LOGIT
        f.create_dataset("pinned_in_training", data=pinned)
        f.create_dataset("transcript_id", data=np.array([x["isoform_id"] for x in recs], dtype="S"))
        f.create_dataset("gene_id", data=np.array([x["gene"] for x in recs], dtype="S"))
        f.create_dataset("arm", data=np.array([x["arm"] for x in recs], dtype="S"))
        f.create_dataset("split", data=np.array([x["split"] for x in recs], dtype="S"))
        f.create_dataset("spans", data=np.array(spans_rows, np.int32))
        f.create_dataset("cand_offset", data=cand_offset)
        f.create_dataset("cand_count", data=cand_count)
        f.attrs["batch_shape_offset"] = floor
        f.attrs["batch_shape_offset_positions"] = int(n_floor_samples)
        f.attrs["checkpoint"] = str(args.checkpoint)
        f.attrs["pool_sha256"] = pool_sha
        f.attrs["perm_draws"] = R
        if spreads:
            f.attrs["perm_spread_effect_sd"] = float(np.mean([x[0] for x in spreads]))
            f.attrs["perm_mean_abs_effect"] = float(np.mean([x[1] for x in spreads]))
            f.attrs["perm_unperturbed_logit_sd"] = float(np.mean([x[2] for x in spreads]))
        f.attrs["split_file"] = str(args.split)
        f.attrs["plan"] = "analysis_plans/RETRAIN_PLAN_2026-08-01.md §9"
        f.attrs["config"] = json.dumps({k: ckargs[k] for k in
                                        ("variant", "conv_channels", "n_bins",
                                         "permute_bins", "blank_sequence",
                                         "blank_junctions", "seed")})
        f.attrs["vals_meaning"] = (
            "vals[i,p,b] = logit(transcript i with the base at 1-based transcript "
            "position p+1 replaced by ACGT[b], in every candidate window "
            "containing that coordinate) - base_logit[i]. NaN where valid is "
            "False and at the observed base.")
        # TWO CONVENTIONS IN ONE FILE, AND THEY DIFFER BY ONE. The array axis of
        # vals/valid/obs is 0-based; spans are 1-based inclusive transcript
        # coordinates, which is what the plan and every other table in this
        # project use. Stated rather than left to be inferred: this project has
        # already lost an hour to a coordinate convention it reconstructed.
        f.attrs["coordinates"] = (
            "vals/valid/obs axis 1 is 0-BASED: index p is 1-based transcript "
            "position p+1. spans are 1-BASED INCLUSIVE: a span [lo, hi] covers "
            "array indices [lo-1, hi-1] inclusive, i.e. vals[i, lo-1:hi]. "
            "spans columns: (transcript row, slot, atg_lo, atg_hi, stop_lo, "
            "stop_hi).")
        f.attrs["logit_definition"] = (
            "log-odds of P(NMD) computed from log P(NMD) directly in float64, "
            "WITHOUT the [1e-6, 1-1e-6] clamp the training path applies. At the "
            "clamp the training logit is constant and every substitution returns "
            "exactly 0.0; base_logit_training and pinned_in_training record which "
            "transcripts the training output would have pinned. That flag covers "
            "the clamp only: near the clamp the training path also COMPRESSES "
            "(0.26x to 0.93x at z_d -16 to -14, all flagged), and above it the "
            "float32 round-trip through P is wrong by 1.12x to 1.44x while the "
            "flag is correctly false. Difference base_logit against "
            "base_logit_training to see the disagreement rather than reading it "
            "off the flag.")

    dt = time.time() - t0
    print(f"\nwrote {outp}  ({outp.stat().st_size/1e6:,.0f} MB)")
    if device == "cuda":
        print(f"  peak GPU memory {torch.cuda.max_memory_allocated()/1e9:.2f} GB "
              f"at chunk_rows={chunk_rows}")
    print(f"  n {N:,}   W {W:,}   valid positions {int(valid.sum()):,}")
    print(f"  transcripts the TRAINING clamp would pin (vals are unaffected): "
          f"{int(pinned.sum()):,} of {N:,} ({100*pinned.mean():.1f}%)")
    print(f"  batch-shape offset REMOVED by the same-chunk baseline: max {floor:.3e} "
          f"over {n_floor_samples:,} positions in {len(recs):,} transcripts")
    print(f"    (this is how far a chunk's no-op sits from the batch-K base pass. "
          f"It is\n     the systematic, K-correlated error the baseline cancels; "
          f"vals do not carry it.)")

    # DOES THIS BANK CARRY ANYTHING? vals are now differences taken inside one
    # batch shape, so the no-op is zero by construction and is no longer a
    # measurement. The batch-shape offset is still the right SCALE to read
    # against: it is the size of difference this pipeline's float32 encoders
    # produce from a change that is not a substitution at all. An effect below it
    # is not distinguishable from arithmetic, and a bank whose effects sit there
    # says nothing however well its geometry checks out.
    fin = np.isfinite(vals)
    av = np.abs(vals[fin])
    print(f"  |effect| vs that offset, over {int(fin.sum()):,} substitutions:")
    print(f"    median {np.median(av):.3e}   p99 {np.percentile(av, 99):.3e}   "
          f"max {av.max():.3e}")
    if floor > 0:
        print(f"    above it: {100*(av > floor).mean():.1f}%   "
              f"above 10x it: {100*(av > 10*floor).mean():.1f}%")
    else:
        print(f"    the offset is exactly zero on this run: "
              f"{100*(av > 0).mean():.1f}% of effects are non-zero")
    if spreads:
        sd, eff, sdb = (np.mean([x[i] for x in spreads]) for i in range(3))
        print(f"  control arm, {R} paired permutation draws:")
        print(f"    mean |effect|                              {eff:.4e}")
        print(f"    sd of the effect across paired draws       {sd:.4e}  "
              f"({sd/max(eff,1e-30):.2f} x the effect)")
        print(f"    sd of the UNPERTURBED logit across draws   {sdb:.4e}  "
              f"({sdb/max(eff,1e-30):.2f} x the effect)")
        print(f"    -> the last line is what an UNPAIRED difference would carry as "
              f"noise.\n       Pairing is what removes it; R averages what is left.")
    print(f"  {dt:.0f}s, {3*int(valid.sum())/max(dt,1e-9):,.0f} substitutions/s")


if __name__ == "__main__":
    main()

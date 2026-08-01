#!/usr/bin/env python
"""
probe_junction_ablation2.py — the same ablations, on a scale that survives them.

WHY A SECOND PASS. probe_junction_ablation.py answered the question it was asked:
zeroing the junction channel costs the reference-start capture gap about a tenth
of its size, so the gap is not junction geometry. But two of its controls make
its OTHER numbers unreadable, and both failures are on this project's own list of
traps.

  1. A DIFFERENCE IN MEANS IS NOT COMPARABLE ACROSS THE CONDITIONS. Zeroing the
     phase channels moves mean p_capture from 0.2498 to 0.1064 and the median the
     other way, 0.0393 to 0.0532: the whole distribution compresses toward the
     middle. A gap measured in probability points shrinks when the scale shrinks,
     whether or not the ordering changed. "Percentage points are not comparable
     across groups with different base rates" is trap 2 in the method document
     and this is the same error one level up, across conditions rather than
     groups.

     FIX: report AUC -- P(a reference start outranks a random non-reference
     candidate). It is invariant to any monotone rescaling, so global compression
     cannot move it, and 0.5 is a real null rather than an assumed one.

  2. THE FILL STRATIFICATION WAS TOO COARSE TO CARRY THE WEIGHT IT WAS GIVEN.
     Upstream fill piles up at 1.0, so quartile edges collapsed to two bands. Two
     bands cannot rule out a confound that ranges over the whole unit interval,
     and this is now the load-bearing control: channels 6-8 hold a period-3
     pattern that is a fixed function of position, IDENTICAL for every candidate,
     written only where the window is filled. Their ONLY candidate-specific
     content is the fill mask. Zeroing them collapsed the gap by 72%, which reads
     as "padding extent carries the gap" -- but zeroing three of nine channels is
     also far off the data manifold, so disruption and information-removal are
     confounded.

     FIX: deciles of upstream fill, and a condition that removes the fill mask
     from channels 6-8 while KEEPING the period-3 carrier (`phase_nofill`), which
     separates "the carrier was destroyed" from "the fill mask was removed".

CACHING. Capture is written to an npz so any later re-cut is free rather than a
re-run, which is how measure_position_bank.py already treats its sweep.

IN-SAMPLE, on chr21, a training chromosome. Reconnaissance.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from model_v6 import ScanningNMDModel          # noqa: E402
from tensor_io import decode_windows           # noqa: E402


def load(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    a = ck["args"]
    m = ScanningNMDModel(conv_channels=a["conv_channels"], n_bins=a["n_bins"],
                         n_structural=1, permute_bins=False)
    m.load_state_dict(ck["model"])
    m.eval()
    assert not bool(a.get("permute_bins", False))
    return m


def capture(model, win, bs=512):
    out = np.empty(len(win), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(win), bs):
            z = model.init_head(model.enc_init(torch.as_tensor(win[i:i + bs])))
            out[i:i + bs] = torch.sigmoid(z.squeeze(-1)).double().numpy()
    return out


def auc(score, pos):
    """P(a positive outranks a negative), ties counted as half. Rank form."""
    pos = np.asarray(pos, bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(score).rank().to_numpy()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def stratified(score, pos, strat, n_min=8, weight="pos"):
    """AUC inside each stratum, weighted by positives. Returns (value, n, thin)."""
    num = den = 0.0
    thin = []
    for s in np.unique(strat):
        m = strat == s
        a, b = m & pos, m & ~pos
        if a.sum() >= n_min and b.sum() >= n_min:
            w = a.sum() if weight == "pos" else m.sum()
            num += w * auc(score[m], a[m])
            den += w
        elif m.any() and a.sum() > 0:
            thin.append((s, int(a.sum()), int(b.sum())))
    return (num / den if den else float("nan")), int(den), thin


def main():
    sys.stdout.reconfigure(line_buffering=True)
    tensor = REPO / "results_tensor_chr21" / "nmd_tensor.h5"
    ckdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    ckpts = sorted(ckdir.glob("b8_s*.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {ckdir}")

    with h5py.File(tensor, "r") as f:
        iso = np.array([s.decode() for s in f["isoform_id"][:]])
        cnt = f["count"][:]
        o_s = f["orf_start"][:]
        codes = f["codes"][:]
        attrs = dict(f.attrs)

    pool = pd.read_csv(REPO / "results_pool_v6" / "orf_pool.tsv", sep="\t",
                       usecols=["isoform_id", "slot", "is_ref_cds", "kozak_score"])
    pool = pool[pool.isoform_id.isin(set(iso))]
    key = {s: i for i, s in enumerate(iso)}
    pool["tx"] = pool.isoform_id.map(key)
    pool = pool.sort_values(["tx", "slot"], kind="stable").reset_index(drop=True)
    assert np.array_equal(pool["tx"].to_numpy(), np.repeat(np.arange(len(iso)), cnt))

    ref = pool["is_ref_cds"].to_numpy() == 1
    slot = pool["slot"].to_numpy()
    kz = pool["kozak_score"].to_numpy()

    anchor = o_s.astype(np.int64)
    L = int(attrs["atg_left"])
    win = decode_windows(codes[:, 0], anchor, L, anchor)

    filled = win[:, 6:9].sum(1) > 0
    up_fill = filled[:, :L].mean(1)

    # channels 6-8 hold (k - left) mod 3, gated by fill. Assert the "identical for
    # every candidate" claim rather than inheriting it: the carrier must be a
    # function of position alone wherever the window is filled.
    k = np.arange(win.shape[2])
    carrier = np.zeros((3, win.shape[2]), dtype=np.float32)
    carrier[(k - L) % 3, k] = 1.0
    assert np.array_equal(win[:, 6:9], carrier[None] * filled[:, None, :]), \
        "channels 6-8 are not a fixed position-only carrier gated by fill"
    print("VERIFIED: channels 6-8 == fixed period-3 carrier x fill mask, so their")
    print("          only candidate-specific content is WHERE THE WINDOW IS FILLED.\n")

    conds = {}
    conds["full"] = win
    for nm, ch in (("zero_junc", [4]), ("zero_gc", [5]),
                   ("zero_phase", [6, 7, 8]), ("zero_seq", [0, 1, 2, 3])):
        w = win.copy(); w[:, ch, :] = 0.0; conds[nm] = w
    w = win.copy()
    w[:, 6:9, :] = carrier[None]          # carrier everywhere; fill mask removed
    conds["phase_nofill"] = w

    print(f"chr21: {len(iso):,} transcripts, {len(pool):,} candidates, "
          f"{len(ckpts)} seeds; ATG window {L}+{attrs['atg_right']}")
    print(f"reference starts {int(ref.sum()):,}; chr21 is a TRAINING chromosome\n")

    cache, rows = {}, []
    for cname, w in conds.items():
        per_seed = np.stack([capture(load(cp), w) for cp in ckpts])
        cap = per_seed.mean(0)
        cache[cname] = per_seed
        moved = np.abs(cap - cache["full"].mean(0))
        a_raw = auc(cap, ref)
        a_slot, n_slot, _ = stratified(cap, ref, slot)
        okk = np.isfinite(kz)
        dec = np.full(len(cap), -1)
        dec[np.where(okk)[0]] = np.asarray(pd.qcut(kz[okk], 10, labels=False, duplicates="drop"))
        a_kz, n_kz, thin_kz = stratified(cap, ref, dec)
        fdec = np.asarray(pd.qcut(up_fill, 10, labels=False, duplicates="drop"))
        a_fill, n_fill, thin_f = stratified(cap, ref, fdec)
        rows.append((cname, cap[ref].mean() - cap[~ref].mean(), a_raw, a_slot,
                     a_kz, a_fill, cap.mean(),
                     float(np.mean(moved > 0)), float(np.median(moved))))
        print(f"=== {cname} ===")
        if cname != "full":
            print(f"  liveness {100*np.mean(moved>0):5.1f}% moved, "
                  f"median |delta| {np.median(moved):.5f}, max {moved.max():.4f}")
        print(f"  mean p_capture {cap.mean():.4f}   median {np.median(cap):.4f}")
        print(f"  AUC ref-vs-other        raw {a_raw:.4f}")
        print(f"    within slot           {a_slot:.4f}  (n_ref {n_slot})")
        print(f"    within Kozak decile   {a_kz:.4f}  (n_ref {n_kz})"
              + (f"  thin: {thin_kz}" if thin_kz else ""))
        print(f"    within up-fill decile {a_fill:.4f}  (n_ref {n_fill})"
              + (f"  thin: {thin_f}" if thin_f else ""))
        print()

    print("=== THE COMPARISON — AUC is the column that means the same thing "
          "in every row ===")
    print(f"  {'condition':<13} {'mean diff':>10} {'AUC raw':>8} {'AUC|slot':>9} "
          f"{'AUC|kozak':>10} {'AUC|fill':>9} {'mean p':>8} {'moved':>7}")
    for (c, md, ar, asl, akz, af, mp, mv, _) in rows:
        print(f"  {c:<13} {md:>+10.4f} {ar:>8.4f} {asl:>9.4f} {akz:>10.4f} "
              f"{af:>9.4f} {mp:>8.4f} {100*mv:>6.1f}%")

    base = rows[0]
    print(f"\n  retained share of the raw AUC's distance above 0.5:")
    for (c, _, ar, *_r) in rows:
        print(f"    {c:<13} {(ar-0.5)/(base[2]-0.5):>7.1%}")

    print("\n=== does capture read padding extent directly? ===")
    cap = cache["full"].mean(0)
    for nm, m in (("all candidates", np.ones(len(cap), bool)),
                  ("non-reference only", ~ref),
                  ("reference only", ref)):
        r = pd.Series(up_fill[m]).corr(pd.Series(cap[m]), method="spearman")
        print(f"  Spearman(upstream fill, p_capture) {nm:<20} {r:+.4f}  "
              f"n {int(m.sum()):,}")
    print("  up-fill decile profile, non-reference candidates only:")
    fdec = np.asarray(pd.qcut(up_fill, 10, labels=False, duplicates="drop"))
    for d in np.unique(fdec):
        m = (fdec == d) & ~ref
        if m.sum():
            print(f"    decile {d}: up_fill {up_fill[m].mean():.3f}   "
                  f"p_capture {cap[m].mean():.4f}   n {int(m.sum()):,}")

    print("\n=== per-seed AUC, so no row rests on one checkpoint ===")
    hdr = "  {:<13}".format("condition") + "".join(
        f"{c.name.replace('.pt',''):>10}" for c in ckpts)
    print(hdr)
    for cname in conds:
        vals = [auc(cache[cname][j], ref) for j in range(len(ckpts))]
        print(f"  {cname:<13}" + "".join(f"{v:>10.4f}" for v in vals))

    out = REPO / "results_interp_all" / "junction_ablation_capture_chr21.npz"
    np.savez_compressed(out, ref=ref, slot=slot, kozak=kz, up_fill=up_fill,
                        seeds=np.array([c.name for c in ckpts]),
                        **{k: v for k, v in cache.items()})
    print(f"\ncached capture -> {out}")


if __name__ == "__main__":
    main()

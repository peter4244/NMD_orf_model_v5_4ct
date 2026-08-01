#!/usr/bin/env python
"""
probe_junction_ablation.py — is the reference-start capture preference SEQUENCE,
or is it WINDOW GEOMETRY?

THE QUESTION. probe_selection_depth.py measured p_capture 0.676 at annotated
starts against 0.234 elsewhere, and showed the gap is not the Kozak PWM. But
capture reads a nine-channel window, and only channels 0-3 are sequence identity.
If the gap rides on channel 4 (the splice-junction mark) or on how much of the
window is padding, then "the model finds the annotated start" is a statement
about exon structure and transcript position, not about initiation.

WHAT THE CHANNELS ARE (tensor_io.decode_windows, verified by reading it):
    0-3  one-hot A,C,G,T          sequence identity
    4    splice-junction mark     junc = (codes >> 3) & 1
    5    rolling GC over 50 nt    derived from sequence, over the FILLED range
    6-8  codon phase              frame = (k - left) mod 3, written only where
                                  FILLED. (k - left) does not depend on the
                                  candidate, so the pattern is identical for
                                  every ORF -- what differs between candidates is
                                  only WHERE IT STOPS, i.e. the fill mask.

WHY GEOMETRY IS A LIVE CONFOUND AND NOT A HYPOTHETICAL. The ATG window is
atg_left=900 upstream by atg_right=100 downstream (tensor attrs). An annotated
start is typically near the 5' end, so most of its 900 nt of upstream context
falls off the transcript and is stored unfilled. A candidate deep in the
transcript has a full window. Padding extent is therefore a near-direct readout
of distance-to-5'-end, it is visible in channels 0-3 AND 6-8, and zeroing
channel 4 does not touch it. Any test of "is it sequence?" that only zeros
channel 4 cannot separate sequence from that.

THE CONDITIONS. Each zeros a channel group in the decoded ATG window and nothing
else. Capture reads the ATG window alone through enc_init + init_head
(model_v6.py:161, and the file's own self-check asserts capture does not move
when the stop window is replaced), so no other input can leak in.

    full        all nine channels                  the baseline to reproduce
    zero_junc   channel 4                          THE TEST
    zero_gc     channel 5                          another single derived channel
    zero_phase  channels 6-8                       cannot carry candidate identity
                                                   except through the fill mask
    zero_seq    channels 0-3                       sequence identity removed,
                                                   junction + GC + phase retained

READING IT. A zero is a claim, not an observation (docs/RULES, and it has cost
this project results). So every condition reports how many candidates it actually
moved. An ablation that changes nothing has not shown a channel is unused; it has
shown the ablation did nothing.

IN-SAMPLE. results_tensor_chr21 is the only tensor built, and chr21 is a TRAINING
chromosome for these checkpoints. Everything here is reconnaissance.
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

CONDITIONS = {
    "full":       None,
    "zero_junc":  [4],
    "zero_gc":    [5],
    "zero_phase": [6, 7, 8],
    "zero_seq":   [0, 1, 2, 3],
}


def load(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    a = ck["args"]
    m = ScanningNMDModel(conv_channels=a["conv_channels"], n_bins=a["n_bins"],
                         n_structural=1,
                         permute_bins=bool(a.get("permute_bins", False)))
    m.load_state_dict(ck["model"])
    m.eval()
    assert not m.training
    assert not bool(a.get("permute_bins", False)), \
        "permute_bins redraws per pass; capture would not be reproducible"
    return m, a


def capture(model, win, bs=512):
    """p_capture for every row of `win` (n, 9, W), through enc_init + init_head.

    This is the whole capture path. Written out rather than calling forward()
    because forward() also needs the stop window and the structural block, and
    the point of the experiment is that capture cannot see them.
    """
    out = np.empty(len(win), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(win), bs):
            a = torch.as_tensor(win[i:i + bs])
            z = model.init_head(model.enc_init(a)).squeeze(-1)
            out[i:i + bs] = torch.sigmoid(z).double().numpy()
    return out


def gap_table(cap, ref, slot, kz, label, n_min=8):
    """Reference-vs-other capture gap, three ways. Returns the matched estimate.

    Deciles are reported in full. A stratum too thin to estimate is printed with
    its n and excluded from the weighted mean rather than dropped silently --
    probe_selection_depth's own table is missing decile 1 with no note, and the
    code that produced that table is not in the repository.
    """
    raw = cap[ref].mean() - cap[~ref].mean()
    print(f"  [{label}] raw gap  {raw:+.4f}   "
          f"ref {cap[ref].mean():.4f} (n={int(ref.sum()):,})   "
          f"other {cap[~ref].mean():.4f} (n={int((~ref).sum()):,})")

    num = den = 0.0
    for k in range(6):
        m = slot == k
        a, b = m & ref, m & ~ref
        if a.sum() >= 20 and b.sum() >= 20:
            d = cap[a].mean() - cap[b].mean()
            num += a.sum() * d
            den += a.sum()
    slot_matched = num / den if den else float("nan")
    print(f"  [{label}] slot-matched (n_ref weighted, slots 0-5, >=20/arm) "
          f"{slot_matched:+.4f}   over n_ref {int(den):,}")

    ok = np.isfinite(kz)
    d10 = pd.qcut(kz[ok], 10, labels=False, duplicates="drop")
    dec = np.full(len(cap), -1)
    dec[np.where(ok)[0]] = d10
    num = den = 0.0
    thin = []
    for i in range(10):
        m = dec == i
        a, b = m & ref, m & ~ref
        if a.sum() >= n_min and b.sum() >= n_min:
            d = cap[a].mean() - cap[b].mean()
            num += a.sum() * d
            den += a.sum()
        elif m.any():
            thin.append((i, int(a.sum()), int(b.sum())))
    kozak_matched = num / den if den else float("nan")
    print(f"  [{label}] Kozak-decile-matched (n_ref weighted, >={n_min}/arm) "
          f"{kozak_matched:+.4f}   over n_ref {int(den):,} of {int(ref.sum()):,}")
    if thin:
        for i, na, nb in thin:
            print(f"        decile {i} EXCLUDED: n_ref {na}, n_other {nb} "
                  f"(below {n_min})")
    return raw, slot_matched, kozak_matched


def main():
    sys.stdout.reconfigure(line_buffering=True)
    tensor = REPO / "results_tensor_chr21" / "nmd_tensor.h5"
    ckdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    ckpts = sorted(ckdir.glob("b8_s*.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {ckdir}")

    with h5py.File(tensor, "r") as f:
        iso = np.array([s.decode() for s in f["isoform_id"][:]])
        off, cnt = f["offset"][:], f["count"][:]
        o_s = f["orf_start"][:]
        codes = f["codes"][:]
        attrs = dict(f.attrs)

    pool = pd.read_csv(REPO / "results_pool_v6" / "orf_pool.tsv", sep="\t",
                       usecols=["isoform_id", "slot", "is_ref_cds", "kozak_score",
                                "admitted_by"])
    pool = pool[pool.isoform_id.isin(set(iso))]
    key = {s: i for i, s in enumerate(iso)}
    pool["tx"] = pool.isoform_id.map(key)
    pool = pool.sort_values(["tx", "slot"], kind="stable").reset_index(drop=True)

    # The pool rows must line up with the tensor's candidate rows, in order.
    # probe_selection_depth asserts only the length; assert the identity too.
    exp_tx = np.repeat(np.arange(len(iso)), cnt)
    assert len(pool) == len(exp_tx), f"{len(pool)} pool rows vs {len(exp_tx)}"
    assert np.array_equal(pool["tx"].to_numpy(), exp_tx), \
        "pool row order does not match the tensor's (transcript, slot) order"

    print(f"chr21: {len(iso):,} transcripts, {len(pool):,} candidates, "
          f"{len(ckpts)} checkpoints from {ckdir}")
    print(f"ATG window: atg_left={attrs['atg_left']} atg_right={attrs['atg_right']} "
          f"W={attrs['window']}")
    print("chr21 is a TRAINING CHROMOSOME for these checkpoints. In-sample.\n")

    ref = pool["is_ref_cds"].to_numpy() == 1
    slot = pool["slot"].to_numpy()
    kz = pool["kozak_score"].to_numpy()
    adm = pool["admitted_by"].to_numpy()

    # ---- decode once, in tensor row order -----------------------------------
    anchor = o_s.astype(np.int64)
    win = decode_windows(codes[:, 0], anchor, int(attrs["atg_left"]), anchor)
    assert win.shape == (len(pool), 9, int(attrs["window"])), win.shape

    # Decoding all rows at once must equal decoding one transcript at a time,
    # which is what probe_selection_depth did. Checked, not assumed.
    i = 3
    sl = slice(int(off[i]), int(off[i]) + int(cnt[i]))
    s0 = o_s[sl].astype(np.int64)
    assert np.array_equal(win[sl], decode_windows(codes[sl][:, 0], s0, 900, s0)), \
        "global decode differs from per-transcript decode"

    # ---- what could geometry carry? model-free ------------------------------
    filled = win[:, 6:9].sum(1) > 0            # phase is written exactly where filled
    fill_frac = filled.mean(1)
    njunc = win[:, 4].sum(1)
    up = slice(0, int(attrs["atg_left"]))      # the 900 nt upstream half
    up_fill = filled[:, up].mean(1)

    print("=== 0. what geometry alone says, before any model ===")
    print("  the two channels-groups that are not sequence identity, ref vs other:")
    for nm, v in (("junction marks in window", njunc),
                  ("fill fraction (whole window)", fill_frac),
                  ("fill fraction (900nt upstream)", up_fill)):
        print(f"    {nm:<32} ref {v[ref].mean():8.4f}   other {v[~ref].mean():8.4f}"
              f"   diff {v[ref].mean()-v[~ref].mean():+8.4f}")
    print(f"    windows with NO junction mark : ref {100*np.mean(njunc[ref]==0):.1f}%"
          f"   other {100*np.mean(njunc[~ref]==0):.1f}%")
    print(f"    upstream fully filled (==1.0) : ref {100*np.mean(up_fill[ref]==1):.1f}%"
          f"   other {100*np.mean(up_fill[~ref]==1):.1f}%")
    print(f"  admitted_by among reference candidates: "
          f"{dict(pd.Series(adm[ref]).value_counts())}")
    print()

    # ---- capture under each condition ---------------------------------------
    results = {}
    base_percand = None
    for cname, chans in CONDITIONS.items():
        w = win if chans is None else win.copy()
        if chans is not None:
            w[:, chans, :] = 0.0
        per_seed = []
        for cp in ckpts:
            model, _ = load(cp)
            per_seed.append(capture(model, w))
        cap = np.mean(per_seed, axis=0)
        results[cname] = (cap, per_seed)
        if cname == "full":
            base_percand = cap
        print(f"=== {cname} ===")
        moved = np.abs(cap - base_percand)
        if cname == "full":
            print("  (baseline)")
        else:
            print(f"  LIVENESS: {100*np.mean(moved > 0):.1f}% of candidates moved at "
                  f"all; {100*np.mean(moved > 1e-3):.1f}% moved > 1e-3; "
                  f"median |delta| {np.median(moved):.5f}; max {moved.max():.4f}")
        print(f"  p_capture mean {cap.mean():.4f}  median {np.median(cap):.4f}")
        gap_table(cap, ref, slot, kz, cname)
        print()

    # ---- the headline comparison --------------------------------------------
    print("=== THE COMPARISON ===")
    print(f"  {'condition':<12} {'raw':>9} {'slot-matched':>14} {'kozak-matched':>15}")
    for cname in CONDITIONS:
        cap = results[cname][0]
        r, s, k = (cap[ref].mean() - cap[~ref].mean()), None, None
        num = den = 0.0
        for kk in range(6):
            m = slot == kk
            a, b = m & ref, m & ~ref
            if a.sum() >= 20 and b.sum() >= 20:
                num += a.sum() * (cap[a].mean() - cap[b].mean()); den += a.sum()
        s = num / den if den else float("nan")
        ok = np.isfinite(kz)
        d10 = pd.qcut(kz[ok], 10, labels=False, duplicates="drop")
        dec = np.full(len(cap), -1); dec[np.where(ok)[0]] = d10
        num = den = 0.0
        for i2 in range(10):
            m = dec == i2
            a, b = m & ref, m & ~ref
            if a.sum() >= 8 and b.sum() >= 8:
                num += a.sum() * (cap[a].mean() - cap[b].mean()); den += a.sum()
        k = num / den if den else float("nan")
        print(f"  {cname:<12} {r:>+9.4f} {s:>+14.4f} {k:>+15.4f}")

    # ---- the geometry the ablation does not remove --------------------------
    print("\n=== 5. the gap inside strata of upstream fill ===")
    print("  zeroing channel 4 leaves padding extent intact, and padding extent")
    print("  is close to distance-to-5'-end. If the gap lives there it survives")
    print("  the junction ablation while still not being a sequence finding.")
    edges = np.quantile(up_fill, [0, .25, .5, .75, 1.0])
    edges = np.unique(edges)
    band = np.clip(np.digitize(up_fill, edges[1:-1]), 0, len(edges) - 2)
    for cname in ("full", "zero_junc"):
        cap = results[cname][0]
        print(f"  --- {cname} ---")
        num = den = 0.0
        for b in range(len(edges) - 1):
            m = band == b
            a, o = m & ref, m & ~ref
            lab = f"up_fill {edges[b]:.2f}-{edges[b+1]:.2f}"
            if a.sum() >= 8 and o.sum() >= 8:
                d = cap[a].mean() - cap[o].mean()
                num += a.sum() * d; den += a.sum()
                print(f"    {lab:<24} ref {cap[a].mean():.4f} (n={int(a.sum()):>3})"
                      f"   other {cap[o].mean():.4f} (n={int(o.sum()):>4})"
                      f"   diff {d:+.4f}")
            else:
                print(f"    {lab:<24} EXCLUDED n_ref {int(a.sum())} "
                      f"n_other {int(o.sum())}")
        print(f"    fill-matched (n_ref weighted): "
              f"{num/den if den else float('nan'):+.4f}   over n_ref {int(den):,}")

    # ---- per-seed spread on the headline ------------------------------------
    print("\n=== 6. per-seed, so the answer is not one checkpoint ===")
    print(f"  {'seed':<12} {'full raw':>10} {'zero_junc raw':>15} {'delta':>9}")
    for j, cp in enumerate(ckpts):
        cf = results["full"][1][j]
        cj = results["zero_junc"][1][j]
        gf = cf[ref].mean() - cf[~ref].mean()
        gj = cj[ref].mean() - cj[~ref].mean()
        print(f"  {cp.name:<12} {gf:>+10.4f} {gj:>+15.4f} {gj-gf:>+9.4f}")


if __name__ == "__main__":
    main()

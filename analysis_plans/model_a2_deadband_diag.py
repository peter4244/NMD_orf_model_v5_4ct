"""
model_a2_deadband_diag.py — what is the dead band actually made of?

The SEQ-A2 primary returned keto 1.081 in the dead band, 3' of the stop, CI
[1.056, 1.106] against a null 95th percentile of 1.017. The dead band is the
instrumental control: nothing there can move the output, so a keto signature
there means magnitude and composition correlate through the ENCODER rather than
through the decay head -- which would bound how much of the live 3' positive
(1.09-1.15) can be attributed to the head at all.

Before that reading is taken seriously, three facts have to be measured rather
than assumed. The incoming interpretability window predicted the first and it is
the reason this script exists.

  1. ARE DEAD `e` VALUES EXACTLY ZERO, or merely tiny? 5.5 says the dead rate
     tracks a float64 resolution boundary exactly, which is consistent with exact
     cancellation. If exact, a top-k cut sits AT zero, every position ties, the
     elevated set is the whole cell, and `cell_ratio`'s k>=n guard drops it. If
     merely tiny, the dead band is a magnitude selection among numerical noise --
     a different mechanism with the same consequence for interpretation.

  2. HOW MANY CELLS DROP, AND WHICH? Measured from the primary log, 78 of 567
     qualifying dead cells 3' of the stop produced no statistic (13.8%), against
     6 of 283 in-ORF. The gate reported the survivors and never said so. The
     dropped cells are the most tie-degenerate, so the survivors are selected
     toward cells with more distinct values -- selection on the very quantity the
     control is measuring.

  3. IS THE DEAD-BAND SIGNAL FLAT IN MASS, OR RISING TOWARD THE CUT? This is what
     separates the two readings and neither the gate nor the row tests it:
       flat   -> an encoder artifact. Composition and numerical magnitude are
                 correlated at positions that cannot respond, so the same
                 correlation is available to inflate the live bands.
       rising -> leakage. 1e-8 is a threshold, not a physical boundary, and
                 positions just below it carry small real effects. Then the dead
                 band is not a clean control and the live positive is not
                 impeached by it -- but the cut needs to move.

Descriptive only. No null, no test, no adjustment. It measures what a control is
made of before that control is used to discount a result.

Namespaced `model_*`; distinct filename so it cannot collide with the running
gate job, which invokes model_a2_gate.py three times from the same path.

Run from the repo root.
"""

import argparse
import numpy as np
import h5py

NT = "ACGT"
DEAD_CUT = 1e-8
KETO = (2, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="results_ism_v6/bank_interp_s100.h5")
    ap.add_argument("--top-frac", type=float, default=0.10)
    ap.add_argument("--floor", type=int, default=100)
    args = ap.parse_args()

    f = h5py.File(args.bank, "r")
    spans = f["spans"][:]
    cand_off = f["cand_offset"][:]
    cand_cnt = f["cand_count"][:]
    p_select = f["p_select"][:]
    orf_start = f["cand_orf_start"][:]
    orf_end = f["cand_orf_end"][:]
    N = len(f["transcript_id"])

    n_exact_zero = n_dead_total = 0
    tie_fracs, cell_rows = [], []
    submass = [[] for _ in range(6)]     # keto counts by decade below the cut
    subn = [[] for _ in range(6)]

    for i in range(N):
        lo, nk = int(cand_off[i]), int(cand_cnt[i])
        b = spans[lo:lo + nk]
        P = int(max(b[:, 3].max(), b[:, 5].max()))
        if P < 50:
            continue
        v = f["vals_decay"][i, :P].astype(np.float64)
        o = f["obs"][i, :P]
        m = f["mass"][i, :P].astype(np.float64)
        with np.errstate(invalid="ignore"):
            e = np.nanmax(np.abs(v), axis=1)
        ok = f["valid"][i, :P].astype(bool) & np.isfinite(e) & (o >= 0)
        idx = np.flatnonzero(ok)
        if not len(idx):
            continue

        ps = p_select[lo:lo + nk]
        k_sel = int(np.argmax(ps))
        s, t = int(orf_start[lo + k_sel]), int(orf_end[lo + k_sel])

        dead = idx[m[idx] < DEAD_CUT]
        if not len(dead):
            continue
        de, dm, do = e[dead], m[dead], o[dead]
        n_dead_total += len(dead)
        n_exact_zero += int((de == 0.0).sum())

        # ---- fact 3: keto fraction of the top decile, by decade of mass below
        # the cut. Computed on all dead positions of the transcript, pooled
        # later, because per-cell n is small.
        with np.errstate(divide="ignore", invalid="ignore"):
            dec = np.floor(np.log10(np.maximum(dm, 1e-300)))
        for j in range(6):
            sel = dec == (-9 - j)          # 1e-9, 1e-10, ... 1e-14
            if sel.sum() >= 20:
                ee, oo = de[sel], do[sel]
                kk = max(1, int(round(args.top_frac * len(ee))))
                cut = np.partition(ee, -kk)[-kk]
                elev = ee >= cut
                submass[j].append(float(np.isin(oo[elev], KETO).mean()))
                subn[j].append(float(np.isin(oo, KETO).mean()))

        # ---- facts 1 and 2, per dead cell, matching the gate's cell definition
        reg = np.full(len(dead), 2, np.int8)
        reg[dead < s] = 0
        reg[(dead >= s) & (dead < t)] = 1
        for ri in range(3):
            sel = reg == ri
            n = int(sel.sum())
            if n < args.floor:
                continue
            ee = de[sel]
            kk = max(1, int(round(args.top_frac * n)))
            cut = np.partition(ee, -kk)[-kk]
            elev = ee >= cut
            k_real = int(elev.sum())
            tie_at_cut = int((ee == cut).sum())
            tie_fracs.append(tie_at_cut / n)
            cell_rows.append((ri, n, kk, k_real, tie_at_cut,
                              int((ee == 0.0).sum()), k_real >= n))
    f.close()

    print(f"BANK {args.bank}   dead cut {DEAD_CUT:g}   top {args.top_frac:.0%}"
          f"   floor >={args.floor}")
    print(f"\nFACT 1 -- ARE DEAD VALUES EXACT ZEROS?")
    print(f"  dead positions              {n_dead_total:,}")
    print(f"  exactly 0.0                 {n_exact_zero:,}"
          f"   ({n_exact_zero/max(1,n_dead_total):.2%})")
    print("  => exact zeros make the top-k cut degenerate: every position ties,")
    print("     the elevated set is the whole cell, and the k>=n guard drops it.")

    rows = np.array([r[:6] for r in cell_rows])
    dropped = np.array([r[6] for r in cell_rows])
    print(f"\nFACT 2 -- HOW MANY DEAD CELLS CANNOT PRODUCE A NUMBER?")
    print(f"  qualifying dead cells       {len(rows):,}")
    print(f"  degenerate (k >= n)         {int(dropped.sum()):,}"
          f"   ({dropped.mean():.1%})")
    print(f"  tie fraction at the cut     deciles "
          + " ".join(f"{np.percentile(tie_fracs, d):.3f}"
                     for d in range(0, 101, 25)))
    for ri, name in enumerate(("5p_of_start", "in_orf", "3p_of_stop")):
        sel = rows[:, 0] == ri
        if sel.sum():
            print(f"    {name:>13}  cells {int(sel.sum()):>5}"
                  f"  degenerate {dropped[sel].mean():>6.1%}"
                  f"  median tie frac {np.median(np.array(tie_fracs)[sel]):.3f}")

    print(f"\nFACT 3 -- IS THE KETO SIGNAL FLAT BELOW THE CUT, OR RISING?")
    print(f"  {'mass decade':>12} {'n_tx':>6} {'keto elev':>10} {'keto bg':>9}"
          f" {'ratio':>7}")
    for j in range(6):
        if len(submass[j]) < 20:
            continue
        a, c = np.mean(submass[j]), np.mean(subn[j])
        print(f"  {'1e-' + str(9 + j):>12} {len(submass[j]):>6} {a:>10.3f}"
              f" {c:>9.3f} {a/c:>7.3f}")
    print("  flat across decades  -> encoder artifact; the same correlation is")
    print("                          available to inflate the live bands")
    print("  rising toward 1e-9   -> leakage of real signal past a threshold that")
    print("                          is numerical, not physical; move the cut")


if __name__ == "__main__":
    main()

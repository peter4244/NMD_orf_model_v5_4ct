#!/usr/bin/env python3
"""
probe_kozak_ablation.py — is the Kozak signal masked by the annotation flags?

THE HYPOTHESIS (Pete, 2026-07-31). The structural branch carries `is_ref_cds` and
`is_sqanti_cds`, which tell the model "this is the annotated start codon" WITHOUT reading the
sequence, and which a sequence substitution does not change. Occlusion and Shapley both measure
a MARGINAL contribution given everything else present, so a redundant non-sequence feature would
suppress the measured Kozak effect even if the model uses initiation context.

THE TEST. Enumerate all 4x4 contexts at Kozak -3 and +4, under two conditions:
    NATIVE  — flags as observed
    ABLATED — both annotation flags set to their normalised-absent value
and compare the -3 purine-vs-pyrimidine and +4 G-vs-notG contrasts.

WHY NO RETRAIN IS NEEDED, and it is worth stating because the obvious objection is that the
model cannot start reading Kozak just because an input was zeroed. This tests whether the model
learned a CONDITIONAL rule — read context when annotation is unavailable — and it is legitimate
only because the ablated state is in the training distribution. Measured: of the 132,823
masked-in ORF slots in the train split, 103,117 (77.6%) already have both flags at zero. The
ablation asks about the majority regime the shared-weight encoder was trained on, not an
extrapolation.

WHAT IT CANNOT CONCLUDE. That a model trained WITHOUT the flags would learn Kozak better. That
is a separate experiment and needs a retrain.

CONFOUNDS HANDLED. Cells where the substitution creates a different motif are excluded from the
primary contrast and reported separately: +4=T creates an in-frame stop codon in ~19% of
isoforms, -3=A creates an in-frame AUG in ~4%.

CAVEAT, stated rather than buried. The substitution is applied to rank 0's window only, not
propagated to the other ORF windows it overlaps. That is the off-manifold version. It is used
here because the quantity of interest is the DIFFERENCE between native and ablated, in which the
contamination is common to both conditions and largely cancels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from model import build_model                                    # noqa: E402
from utils import NMDDataset, resolve_checkpoint                 # noqa: E402

TAG, W = "atg1000_stop1000", 1000
SEEDS = [100, 200, 300, 400, 500]
N_ISO = 300
NT = "ACGT"
GC_WIN = 50
REF_I, SQ_I = 2, 3          # orf_features columns for is_ref_cds / is_sqanti_cds


def rolling_gc(is_gc, window=GC_WIN):
    n = len(is_gc)
    cs = np.concatenate([[0.0], np.cumsum(is_gc)])
    half = window // 2
    p = np.arange(n)
    lo = np.maximum(0, p - half); hi = np.minimum(n, p + half + 1)
    return ((cs[hi] - cs[lo]) / (hi - lo)).astype(np.float32)


def span(win):
    on = win[6:9].sum(axis=0) > 0
    i = np.flatnonzero(on)
    return (int(i[0]), int(i[-1] + 1)) if len(i) else (0, 0)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    cfg = yaml.safe_load((REPO / "config_dn.yaml").read_text())
    h5 = str(REPO / cfg["data"]["hdf5_path"])
    with h5py.File(h5) as f:
        mu = f["normalization/orf_feat_mean"][:].astype(np.float32)
        sd = f["normalization/orf_feat_std"][:].astype(np.float32)
    abl_ref, abl_sq = (0 - mu[REF_I]) / sd[REF_I], (0 - mu[SQ_I]) / sd[SQ_I]
    print(f"normalised-absent flag values: is_ref_cds {abl_ref:+.3f}, is_sqanti_cds {abl_sq:+.3f}")

    ds = NMDDataset(h5, W, W, split="all", restrict_to=np.arange(N_ISO))
    i_m3, i_p4 = W // 2 - 4, W // 2 + 2          # verified: Kozak -3 and +4
    i_p5, i_p6 = W // 2 + 3, W // 2 + 4
    i_m2, i_m1 = W // 2 - 3, W // 2 - 2

    rows = {}
    for seed in SEEDS:
        m = build_model({**cfg["model"], "window_size_atg": W, "window_size_stop": W})
        m.load_state_dict(torch.load(resolve_checkpoint(REPO / "results_4ct_sweep", TAG, seed),
                                     map_location="cpu", weights_only=False)["model_state_dict"])
        m.eval()
        out = {c: {"m3": [], "p4": []} for c in ("native", "ablated")}
        for i in range(N_ISO):
            s = ds[i]
            aw = s["atg_windows"].numpy(); w0 = aw[0]
            lo, hi = span(w0)
            if not (lo <= i_m3 and i_p6 < hi):
                continue
            b = lambda j: NT[int(w0[0:4, j].argmax())]
            creates_stop = (b(i_p5), b(i_p6)) in {("A", "A"), ("A", "G"), ("G", "A")}
            creates_aug = (b(i_m2), b(i_m1)) == ("T", "G")

            wins, cells = [], []
            for a3 in range(4):
                for a4 in range(4):
                    w = w0.copy()
                    w[0:4, i_m3] = 0; w[a3, i_m3] = 1
                    w[0:4, i_p4] = 0; w[a4, i_p4] = 1
                    seg = w[0:4, lo:hi]
                    w[5, lo:hi] = rolling_gc(seg[1] + seg[2])
                    wins.append(w); cells.append((NT[a3], NT[a4]))
            batch = np.repeat(aw[None], 16, axis=0); batch[:, 0] = np.stack(wins)

            for cond in ("native", "ablated"):
                feat = s["orf_features"].clone()
                if cond == "ablated":
                    feat[:, REF_I] = float(abl_ref); feat[:, SQ_I] = float(abl_sq)
                    feat[~s["orf_mask"]] = 0.0
                with torch.no_grad():
                    lg = m(torch.from_numpy(batch),
                           s["stop_windows"].unsqueeze(0).expand(16, -1, -1, -1),
                           feat.unsqueeze(0).expand(16, -1, -1),
                           s["orf_mask"].unsqueeze(0).expand(16, -1)).squeeze(-1).numpy()
                v = {c: lg[k] for k, c in enumerate(cells)}
                pur = [v[(x, y)] for x, y in cells if x in "AG"
                       and not (creates_aug and x == "A") and not (creates_stop and y == "T")]
                pyr = [v[(x, y)] for x, y in cells if x in "CT"
                       and not (creates_stop and y == "T")]
                gg = [v[(x, y)] for x, y in cells if y == "G" and not (creates_aug and x == "A")]
                ng = [v[(x, y)] for x, y in cells if y != "G"
                      and not (creates_stop and y == "T") and not (creates_aug and x == "A")]
                if pur and pyr: out[cond]["m3"].append(np.mean(pur) - np.mean(pyr))
                if gg and ng:   out[cond]["p4"].append(np.mean(gg) - np.mean(ng))
        rows[seed] = {c: {k: float(np.mean(v)) for k, v in d.items()} for c, d in out.items()}
        n = len(out["native"]["m3"])
        print(f"  seed {seed}: n={n}  "
              f"-3 native {rows[seed]['native']['m3']:+.5f} ablated {rows[seed]['ablated']['m3']:+.5f}   "
              f"+4 native {rows[seed]['native']['p4']:+.5f} ablated {rows[seed]['ablated']['p4']:+.5f}")

    print("\n=== KOZAK CONTRASTS, native vs annotation-flags-ablated ===")
    for pos, lab in (("m3", "-3 purine - pyrimidine"), ("p4", "+4 G - notG")):
        nat = np.array([rows[s]["native"][pos] for s in SEEDS])
        abl = np.array([rows[s]["ablated"][pos] for s in SEEDS])
        print(f"\n  {lab}")
        print(f"    native : mean {nat.mean():+.5f}  sd {nat.std(ddof=1):.5f}  "
              f"members {np.array2string(nat, precision=5, floatmode='fixed')}")
        print(f"    ablated: mean {abl.mean():+.5f}  sd {abl.std(ddof=1):.5f}  "
              f"members {np.array2string(abl, precision=5, floatmode='fixed')}")
        print(f"    signs agree: native {'YES' if len(set(np.sign(nat)))==1 else 'NO'}, "
              f"ablated {'YES' if len(set(np.sign(abl)))==1 else 'NO'}")
        r = abs(abl.mean()) / abs(nat.mean()) if nat.mean() else float('nan')
        print(f"    |ablated| / |native| = {r:.2f}x")


if __name__ == "__main__":
    main()

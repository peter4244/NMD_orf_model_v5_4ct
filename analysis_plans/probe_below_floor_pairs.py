#!/usr/bin/env python
"""
probe_below_floor_pairs.py — which population should the matched-pair claim use?

THE DISAGREEMENT, narrow and testable. Maude restricts every arm to
kozak_score >= the MANE floor (-1.2507921188400943), dropping 178 of 1,909 chr2
reference starts, on the ground that non-reference candidates are admitted only
at or above the floor while reference starts bypass it
(`reference_start_always_admitted: true` in the tensor's admission attrs). She
reads my unrestricted population as containing an arm that could not have held a
competitor.

FOR THE MARGINAL COMPARISON SHE IS RIGHT. Comparing all reference starts against
all non-reference candidates does put below-floor references in a Kozak range
where no competitor can exist, so the contrast is asymmetric by construction.

FOR THE MATCHED-PAIR TEST I THINK IT IS BACKWARDS, and this script decides it.
The pair test matches on POSITION, not on Kozak. A below-floor reference paired
with an above-floor competitor at the same position is a perfectly well-defined
comparison -- and it is the HARDEST one available, because the reference is
carrying the worst possible PWM disadvantage. The model cannot see admission
status, cannot see kozak_score, and cannot see is_ref_cds; it sees the window. So
dropping those 178 removes precisely the cases that discriminate "capture reads
the PWM" from "capture reads something else".

If below-floor references still win their position-matched pairs, that is the
strongest anti-PWM evidence in the analysis and it should be reported, not
filtered out. If they lose, Maude's restriction is protecting a real artifact and
I withdraw.

Reports both populations so the manuscript can state which it used.
chr2 is VALIDATION -- held out from gradients, not from model selection.
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

FLOOR = -1.2507921188400943


def load(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    a = ck["args"]
    m = ScanningNMDModel(conv_channels=a["conv_channels"], n_bins=a["n_bins"],
                         n_structural=1, permute_bins=False)
    m.load_state_dict(ck["model"]); m.eval()
    return m


def capture(model, codes, anchor, L, bs=256):
    out = np.empty(len(codes), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(codes), bs):
            w = decode_windows(codes[i:i + bs], anchor[i:i + bs], L, anchor[i:i + bs])
            z = model.init_head(model.enc_init(torch.as_tensor(w)))
            out[i:i + bs] = torch.sigmoid(z.squeeze(-1)).double().numpy()
    return out


def build_pairs(tx, ref, o_s, keep, D=50):
    pairs = []
    for t in np.unique(tx):
        idx = np.where((tx == t) & keep)[0]
        r_i, o_i = idx[ref[idx]], idx[~ref[idx]]
        if not len(r_i) or not len(o_i):
            continue
        for ri in r_i:
            for oi in o_i[np.abs(o_s[o_i] - o_s[ri]) <= D]:
                pairs.append((ri, oi))
    return np.array(pairs) if pairs else np.empty((0, 2), int)


def report(name, P, cap, o_s, kz):
    if not len(P):
        print(f"  {name}: no pairs"); return
    rc, oc = cap[P[:, 0]], cap[P[:, 1]]
    won = rc > oc
    up = o_s[P[:, 0]] < o_s[P[:, 1]]
    print(f"  {name}")
    print(f"    pairs {len(P):>6,}   refs {len(np.unique(P[:,0])):>5,}   "
          f"capture wins {100*won.mean():>5.1f}%")
    for lab, m in (("ref upstream", up), ("ref downstream", ~up)):
        if m.sum():
            print(f"      {lab:<16} {100*won[m].mean():>5.1f}%   n {int(m.sum()):,}")
    dk = kz[P[:, 0]] - kz[P[:, 1]]
    print(f"      ref has worse Kozak: {100*won[dk < -0.25].mean():>5.1f}%   "
          f"n {int((dk < -0.25).sum()):,}")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ckpts = sorted(Path(sys.argv[1]).glob("b8_s*.pt"))

    with h5py.File(REPO / "results_tensor_chr2" / "nmd_tensor.h5", "r") as f:
        iso = np.array([s.decode() for s in f["isoform_id"][:]])
        cnt = f["count"][:]
        o_s = f["orf_start"][:].astype(np.int64)
        codes = f["codes"][:, 0]
        L = int(f.attrs["atg_left"])

    pool = pd.read_csv(REPO / "results_pool_v6" / "orf_pool.tsv", sep="\t",
                       usecols=["isoform_id", "slot", "is_ref_cds", "kozak_score",
                                "admitted_by"])
    pool = pool[pool.isoform_id.isin(set(iso))]
    pool["tx"] = pool.isoform_id.map({s: i for i, s in enumerate(iso)})
    pool = pool.sort_values(["tx", "slot"], kind="stable").reset_index(drop=True)
    assert np.array_equal(pool["tx"].to_numpy(), np.repeat(np.arange(len(iso)), cnt))

    ref = pool["is_ref_cds"].to_numpy() == 1
    kz = pool["kozak_score"].to_numpy()
    tx = pool["tx"].to_numpy()
    adm = pool["admitted_by"].to_numpy()

    below = ref & (kz < FLOOR)
    print(f"chr2: {int(ref.sum()):,} reference starts; "
          f"{int(below.sum()):,} below the MANE floor {FLOOR:.6f}")
    print(f"  {int(ref.sum())} - {int(below.sum())} = {int(ref.sum()-below.sum()):,}"
          f"   (Maude reports 1,731)")
    print(f"  non-reference candidates below the floor: "
          f"{int(((~ref) & (kz < FLOOR)).sum()):,}  <- must be ~0 if the floor binds")
    print(f"  admitted_by for below-floor references: "
          f"{dict(pd.Series(adm[below]).value_counts())}\n")

    cache = REPO / "results_interp_all" / "capture_chr2.npz"
    if cache.exists():
        cap = np.load(cache)["cap"]
        print(f"loaded cached capture from {cache.name}\n")
    else:
        cap = np.stack([capture(load(cp), codes, o_s, L) for cp in ckpts]).mean(0)
        np.savez_compressed(cache, cap=cap)
        print(f"cached capture -> {cache.name}\n")

    print("=== position-matched pairs, D=50, three populations ===")
    report("ALL references (mine)", build_pairs(tx, ref, o_s, np.ones(len(ref), bool)),
           cap, o_s, kz)
    keep_floor = kz >= FLOOR
    report("FLOOR-RESTRICTED (Maude's)", build_pairs(tx, ref, o_s, keep_floor),
           cap, o_s, kz)

    # the sharp cut: reference below the floor, competitor above it
    P = build_pairs(tx, ref, o_s, np.ones(len(ref), bool))
    m = (kz[P[:, 0]] < FLOOR) & (kz[P[:, 1]] >= FLOOR)
    print("\n=== THE SHARP CUT — reference BELOW the floor vs competitor ABOVE it ===")
    print("  the reference carries the worst available PWM disadvantage, is in the")
    print("  pool only because it is the reference, and is position-matched.")
    report("below-floor refs only", P[m], cap, o_s, kz)
    if m.sum():
        dk = kz[P[m][:, 0]] - kz[P[m][:, 1]]
        print(f"    mean Kozak deficit of the reference in these pairs: {dk.mean():+.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Trace the ATF4 case study first-hand — capture AND selection, all candidates.

WHY. The results/figures handoff prints an ATF4 table with a `p_capture` column
and P(select) values (0.572 / 0.278 / 0.022) that appear in no runlog, no script
output and no claim-value row anywhere in either repo. Its only traceable
producer, analysis2_selection_mass_full.py, prints kozak rather than capture and
gives different selection values. ATF4 is slated for its own Figure 5 panel, so
the numbers on it have to come from the model rather than from a document.

Capture is parts["p"] = sigmoid(z_p); selection is parts["p_select"], the
stick-breaking product. Both are averaged over the same five checkpoints the
producer averages, in the same candidate (slot) order.
"""
import sys
from pathlib import Path
import h5py
import numpy as np
import torch

REPO = Path.home() / "claude_projects" / "NMD_orf_model_v5_4ct"
sys.path.insert(0, str(REPO))
from model_v6 import ScanningNMDModel          # noqa: E402
from tensor_io import decode_windows           # noqa: E402

TENSOR = REPO / "results_tensor_v6" / "nmd_tensor.h5"
CKPT = REPO / "results_interp_all" / "ckpt_interp_c32_b8"
TARGETS = {"ENST00000674920.3": 283, "ENST00000337304.2": 883}   # annotated start, per producer

with h5py.File(TENSOR, "r") as f:
    iso = np.array([s.decode() for s in f["isoform_id"][:]])
    split = np.array([s.decode() for s in f["split"][:]])
    off, cnt = f["offset"][:], f["count"][:]
    o_s = f["orf_start"][:].astype(np.int64)
    o_e = f["orf_end"][:].astype(np.int64)
    struct = f["structural"][:]
    codes = f["codes"][:]
    L, SL = int(f.attrs["atg_left"]), int(f.attrs["stop_left"])

ckpts = sorted(CKPT.glob("b8_s*.pt"))
print(f"averaging over {len(ckpts)} checkpoints: {[c.name for c in ckpts]}\n")

for name, c_ann in TARGETS.items():
    hit = np.flatnonzero(iso == name)
    if not len(hit):
        print(f"{name}: not in this tensor\n"); continue
    i = int(hit[0])
    K = int(cnt[i])
    rows = np.arange(off[i], off[i] + K)
    s0, e0 = o_s[rows], o_e[rows]
    atg = decode_windows(codes[rows][:, 0], s0, L, s0)
    stp = decode_windows(codes[rows][:, 1], e0 - 1, SL, s0)
    st = struct[rows][:, [0]]
    W = atg.shape[2]
    A = atg[None].astype(np.float32); S = stp[None].astype(np.float32)
    U = st[None].astype(np.float32); M = np.ones((1, K), bool)

    cap = np.zeros(K); sel = np.zeros(K)
    for cp in ckpts:
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        a = ck["args"]
        m = ScanningNMDModel(conv_channels=a["conv_channels"], n_bins=a["n_bins"],
                             n_structural=1, permute_bins=False)
        m.load_state_dict(ck["model"]); m.eval()
        with torch.no_grad():
            _, parts = m(torch.as_tensor(A), torch.as_tensor(S),
                         torch.as_tensor(U), torch.as_tensor(M), return_parts=True)
        cap += parts["p"].numpy()[0, :K] / len(ckpts)
        sel += parts["p_select"].numpy()[0, :K] / len(ckpts)

    tot = sel.sum()
    print(f"{name}   split={split[i]}   candidates={K}   annotated start={c_ann}")
    print(f"  sum P(select) over candidates = {tot:.4f}  "
          f"(residual mass escaping every candidate = {1-tot:.4f})")
    up = s0 < c_ann
    print(f"  mass upstream {sel[up].sum()/tot:.3f}   "
          f"on overlapping {sel[up & (e0 >= c_ann)].sum()/tot:.3f}   "
          f"on annotated {sel[s0 == c_ann].sum()/tot:.3f}")
    print(f"  {'rank':>4} {'orf_start':>10} {'orf_end':>9} {'len':>6} "
          f"{'p_capture':>10} {'P(select)':>10}  position")
    for r, k in enumerate(np.argsort(-sel), 1):
        posn = ("upstream" if s0[k] < c_ann else
                "annotated" if s0[k] == c_ann else "downstream")
        if s0[k] < c_ann and e0[k] >= c_ann:
            posn = "OVERLAPPING"
        print(f"  {r:>4} {s0[k]:>10,} {e0[k]:>9,} {e0[k]-s0[k]+1:>6,} "
              f"{cap[k]:>10.3f} {sel[k]/tot:>10.3f}  {posn}")
    print(f"  highest capture of any candidate: {cap.max():.3f} at orf_start "
          f"{s0[int(np.argmax(cap))]:,}  "
          f"(annotated start capture = {cap[s0 == c_ann][0]:.3f})")
    print()

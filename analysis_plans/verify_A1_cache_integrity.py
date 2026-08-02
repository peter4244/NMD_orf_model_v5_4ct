#!/usr/bin/env python
"""Validate the cached test logits against a fresh forward pass for one seed.

The A1 metric trace read `test_logits_interp_c32_b8.npz`, which is the output of
someone else's forward pass. This regenerates seed 100's row from the checkpoint
and compares. If row 0 reproduces, the cache is the model's output and the whole
A1 trace stands on the checkpoints rather than on a file.
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
LCACHE = REPO / "results_interp_all" / "test_logits_interp_c32_b8.npz"

with h5py.File(TENSOR, "r") as f:
    split = np.array([s.decode() for s in f["split"][:]])
    labels = f["labels"][:].astype(int)
    off, cnt = f["offset"][:], f["count"][:]
    o_s = f["orf_start"][:].astype(np.int64)
    o_e = f["orf_end"][:].astype(np.int64)
    struct = f["structural"][:]
    codes = f["codes"][:]
    L, SL = int(f.attrs["atg_left"]), int(f.attrs["stop_left"])

te = np.flatnonzero(split == "test")
cached = np.load(LCACHE)["logits"]


def batches(counts, max_padded=2048):
    order = np.argsort(counts, kind="stable")
    out, cur, cmax = [], [], 0
    for i in order:
        k = int(counts[i]); nm = max(cmax, k)
        if cur and (len(cur) + 1) * nm > max_padded:
            out.append(np.array(cur)); cur, cmax = [i], k
        else:
            cur.append(i); cmax = nm
    if cur:
        out.append(np.array(cur))
    return out


ckpts = sorted(CKPT.glob("b8_s*.pt"))
si = 0
cp = ckpts[si]
print(f"regenerating {cp.name} (row {si} of the cache)", flush=True)

ck = torch.load(cp, map_location="cpu", weights_only=False)
a = ck["args"]
print(f"  checkpoint args: conv_channels={a['conv_channels']} n_bins={a['n_bins']} "
      f"epoch={ck.get('epoch')} stored_val_auc={ck.get('val_auc')}", flush=True)
m = ScanningNMDModel(conv_channels=a["conv_channels"], n_bins=a["n_bins"],
                     n_structural=1, permute_bins=False)
m.load_state_dict(ck["model"]); m.eval()

fresh = np.zeros(len(te))
pos = {int(t): j for j, t in enumerate(te)}
bs = batches(cnt[te])
with torch.no_grad():
    for bi, b in enumerate(bs):
        sel = te[b]
        c = cnt[sel].astype(int); K = int(c.max())
        rows = np.concatenate([np.arange(off[i], off[i] + cnt[i]) for i in sel])
        s0, e0 = o_s[rows], o_e[rows]
        atg = decode_windows(codes[rows][:, 0], s0, L, s0)
        stp = decode_windows(codes[rows][:, 1], e0 - 1, SL, s0)
        st = struct[rows][:, [0]]
        W = atg.shape[2]
        A = np.zeros((len(sel), K, 9, W), np.float32); S = np.zeros_like(A)
        U = np.zeros((len(sel), K, 1), np.float32)
        M = np.zeros((len(sel), K), bool)
        at = 0
        for j, cc in enumerate(c):
            A[j, :cc] = atg[at:at+cc]; S[j, :cc] = stp[at:at+cc]
            U[j, :cc] = st[at:at+cc]; M[j, :cc] = True; at += cc
        lg = m(torch.as_tensor(A), torch.as_tensor(S),
               torch.as_tensor(U), torch.as_tensor(M)).numpy()
        for j, i in enumerate(sel):
            fresh[pos[int(i)]] = lg[j]
        if bi % 10 == 0:
            print(f"  batch {bi+1}/{len(bs)}", flush=True)

d = np.abs(fresh - cached[si])
print(f"\nmax |fresh - cached| = {d.max():.3e}   median {np.median(d):.3e}")
print(f"identical to 1e-5: {bool(d.max() < 1e-5)}")

from sklearn.metrics import roc_auc_score, average_precision_score
y = labels[te]
print(f"\nseed 100 from the fresh pass:  AUC {roc_auc_score(y, fresh):.4f}   "
      f"AUPRC {average_precision_score(y, fresh):.4f}")
print(f"seed 100 from the cache:       AUC {roc_auc_score(y, cached[si]):.4f}   "
      f"AUPRC {average_precision_score(y, cached[si]):.4f}")

swap = cached.copy(); swap[si] = fresh
print(f"\nensemble with the fresh row swapped in (mean logit): "
      f"AUC {roc_auc_score(y, swap.mean(0)):.4f}   "
      f"AUPRC {average_precision_score(y, swap.mean(0)):.4f}")
print(f"reported: AUC 0.9327   AUPRC 0.8684")

#!/usr/bin/env python3
"""
infer_uorf_attention.py — Inference for uORF-attention analysis (v5_4ct).

Runs the v5_4ct best checkpoint (ATG=500, STOP=500) over the FULL v5_4ct
labeled universe (~40k isoforms, training+val+test), producing per-isoform
attention weights across the 5 priority ORFs and NMD probability scores.

The full-universe scope is intentional: this is an ATTRIBUTION analysis
(where does attention land?), not a prediction-generalization analysis.
Predictive overfitting concerns apply to AUC on training data, not to
attention-pattern interpretation, so we use all isoforms the model has
labels for.

Inputs (all relative to this script's directory):
  results_4ct/best_model_atg500_stop500.pt   — v5_4ct trained weights
  results_4ct/tx_summary.tsv                  — v5_4ct labels (is_nmd 0/1)
  results_4ct/nmd_orf_data.h5                 — v5 HDF5 inputs (symlinked;
                                                features are unchanged from
                                                v5 per v5_4ct CLAUDE.md)
  config.yaml                                 — model + training config

Normalization stats are HARD-CODED here as the v5_4ct training-set
arrays (extracted from results_4ct/nmd_orf_data.h5 on the cluster, since
the v5 HDF5's stats reflect the v5 universe). This is intentional so the
exact-normalization decision is reviewable in this file rather than
hidden in input data.

Output:
  results_4ct/uorf_attention_predictions.tsv
    Columns: isoform_id, split, chr, label, prob, logit,
             attn_0, attn_1, attn_2, attn_3, attn_4
"""

import os, sys
import numpy as np
import pandas as pd
import h5py
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from model import build_model
from utils import load_config

# ── v5_4ct training-set normalization stats (extracted from cluster HDF5) ──
# Feature order: [frac_start, frac_stop, is_ref_cds, is_sqanti_cds, n_downstream_ejc]
NORM_MEAN = np.array([0.3708552, 0.5446399, 0.13680594, 0.19173051, 1.7330148],
                     dtype=np.float32)
NORM_STD  = np.array([0.26555353, 0.26279053, 0.3435218, 0.39381742, 3.6128473],
                     dtype=np.float32)

# ── Paths ──
CKPT  = os.path.join(REPO, "results_4ct", "best_model_atg500_stop500.pt")
H5    = os.path.join(REPO, "results_4ct", "nmd_orf_data.h5")
TXSUM = os.path.join(REPO, "results_4ct", "tx_summary.tsv")
OUT   = os.path.join(REPO, "results_4ct", "uorf_attention_predictions.tsv")
WS_ATG, WS_STOP = 500, 500
BATCH = 256

# ── Load model ──
config = load_config(os.path.join(REPO, "config.yaml"))
print(f"Loading checkpoint: {CKPT}")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
model_cfg = {**config["model"], "window_size_atg": WS_ATG, "window_size_stop": WS_STOP}
model = build_model(model_cfg)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"  trained {ckpt['epoch']} epochs, val AUC {ckpt.get('val_auc', float('nan')):.4f}")

# ── Define labeled universe ──
# v5_4ct tx_summary.tsv contains ONLY labeled isoforms (NMD=1 or non-NMD=0),
# so the full file is the labeled universe.
tx = pd.read_csv(TXSUM, sep="\t", quoting=3)
tx.columns = [c.strip('"') for c in tx.columns]
for col in tx.select_dtypes(include="object"):
    tx[col] = tx[col].str.strip('"')
labeled = set(tx["isoform_id"])
print(f"v5_4ct labeled universe: {len(labeled)} isoforms "
      f"({(tx['is_nmd']==1).sum()} NMD, {(tx['is_nmd']==0).sum()} non-NMD)")

# ── Stream HDF5 in row order, filtering to labeled universe ──
with h5py.File(H5, "r") as f:
    all_isos = np.array([s.decode() if isinstance(s, bytes) else s
                         for s in f["isoform_id"][:]])
    all_splits = np.array([s.decode() if isinstance(s, bytes) else s
                           for s in f["split"][:]])
    all_chrs = np.array([s.decode() if isinstance(s, bytes) else s
                         for s in f["chr"][:]])
    all_labels_h5 = f["labels"][:].astype(np.int32)

    keep_mask = np.array([iso in labeled for iso in all_isos])
    keep_idx  = np.where(keep_mask)[0]
    print(f"HDF5 contains {len(all_isos)} isoforms; "
          f"{len(keep_idx)} are in v5_4ct labeled universe")

    # batch over kept indices (ascending — h5py requires sorted fancy indexing)
    results = {"isoform_id": [], "split": [], "chr": [], "label": [],
               "prob": [], "logit": [],
               "attn_0": [], "attn_1": [], "attn_2": [], "attn_3": [], "attn_4": []}

    n_done = 0
    for start in range(0, len(keep_idx), BATCH):
        chunk = keep_idx[start:start + BATCH]
        atg  = f[f"w{WS_ATG}/atg_windows"][chunk].astype(np.float32)
        stp  = f[f"w{WS_STOP}/stop_windows"][chunk].astype(np.float32)
        ofs  = f["orf_features"][chunk].astype(np.float32)
        msk  = f["orf_mask"][chunk].astype(bool)

        # v5_4ct normalization (per training-set stats; broadcast over isoforms × ORFs)
        ofs_norm = (ofs - NORM_MEAN[None, None, :]) / NORM_STD[None, None, :]

        # torch.from_numpy is broken on this numpy 2.x / torch combo — go via tolist()
        def T(a, dtype=None):
            if a.dtype == np.bool_:
                return torch.tensor(a.tolist())
            return torch.tensor(a.tolist(), dtype=dtype or torch.float32)

        with torch.no_grad():
            logits, attn = model(T(atg), T(stp), T(ofs_norm), T(msk),
                                  return_attention=True)
        probs   = torch.sigmoid(logits.squeeze(-1)).tolist()
        logitsL = logits.squeeze(-1).tolist()
        attnL   = attn.tolist()

        for j, idx in enumerate(chunk):
            results["isoform_id"].append(all_isos[idx])
            results["split"     ].append(all_splits[idx])
            results["chr"       ].append(all_chrs[idx])
            results["label"     ].append(int(all_labels_h5[idx]))
            results["prob"      ].append(probs[j])
            results["logit"     ].append(logitsL[j])
            for k in range(5):
                results[f"attn_{k}"].append(attnL[j][k])

        n_done += len(chunk)
        if n_done % (BATCH * 16) == 0 or n_done == len(keep_idx):
            print(f"  inferred {n_done}/{len(keep_idx)}")

# ── Save ──
df = pd.DataFrame(results)
df.to_csv(OUT, sep="\t", index=False, float_format="%.6f")
print(f"\nSaved: {OUT}  ({len(df)} rows)")

# Quick sanity summary
print("\nSanity:")
print(f"  splits: {df['split'].value_counts().to_dict()}")
print(f"  label=1 prob mean: {df.loc[df['label']==1, 'prob'].mean():.3f}")
print(f"  label=0 prob mean: {df.loc[df['label']==0, 'prob'].mean():.3f}")

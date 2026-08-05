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

Inputs (all under --results-dir, itself relative to this script's directory):
  <results-dir>/best_model_atg500_stop500[_seed<N>].pt  — v5_4ct trained weights
      (the _seed<N> form is one ensemble member; select it with --member-seed)
  <results-dir>/tx_summary.tsv                — v5_4ct labels (is_nmd 0/1)
  <results-dir>/nmd_orf_data.h5               — v5 HDF5 inputs (symlinked;
                                                features are unchanged from
                                                v5 per v5_4ct CLAUDE.md)
  config.yaml                                 — model + training config

Normalization stats are HARD-CODED here as the v5_4ct training-set arrays,
extracted on the cluster from the PUBLISHED results_4ct/nmd_orf_data.h5 (the
v5 HDF5's own stats reflect the v5 universe). Hardcoding was intentional, so
the exact-normalization decision is reviewable in this file rather than hidden
in input data.

**THEY ARE NOW READ FROM THE HDF5, and the literals are only a fallback.**
data_prep.py writes normalization/orf_feat_{mean,std} into every HDF5 from that
build's own training split, and utils.py -- the path 03_train.py and evaluate.py
use -- reads them from there. This script was the only place in the repo that
bypassed that, so under --results-dir results_4ct_dn the model was TRAINED with
the deposit-native statistics while its inputs here were normalized with the
PUBLISHED ones: a silent train/inference mismatch scaling every feature.

The literals below remain for an HDF5 predating the normalization group, and any
disagreement between the two is PRINTED. An earlier draft of this docstring said
the stats came from "the results dir", which read as though they followed the
flag when they did not; they now do, by reading them where every other consumer
does.

Output:
  <results-dir>/uorf_attention_predictions.tsv
    Columns: isoform_id, split, chr, label, prob, logit,
             attn_0, attn_1, attn_2, attn_3, attn_4

--results-dir MATCHES 03_train.py / evaluate.py / 11_kernel_shap_branches.py /
deepshap.py / 10_export_stop_codon_freq_sf37.py. It is not cosmetic: the
deposit-native rebuild writes to results_4ct_dn, and a hardcoded results_4ct
silently reads the PUBLISHED checkpoint and reports it as deposit-native. This
script had no argument parsing at all, so there was no way to point it anywhere.
"""

import argparse
import os, sys
import numpy as np
import pandas as pd
import h5py
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from model import build_model
from utils import load_config, resolve_checkpoint, selected_tag

# ── v5_4ct training-set normalization stats ──
# EXTRACTED FROM THE PUBLISHED results_4ct HDF5 AND NOT SELECTED BY --results-dir. See the
# module docstring: a deposit-native run normalizes with these published constants, which are
# not the deposit-native universe's true means and sds. W76.
# Feature order: [frac_start, frac_stop, is_ref_cds, is_sqanti_cds, n_downstream_ejc]
NORM_MEAN = np.array([0.3708552, 0.5446399, 0.13680594, 0.19173051, 1.7330148],
                     dtype=np.float32)
NORM_STD  = np.array([0.26555353, 0.26279053, 0.3435218, 0.39381742, 3.6128473],
                     dtype=np.float32)

# ── Paths ──
# results_4ct was hardcoded four times here and this script took no arguments, so there was
# no way to point it at the deposit-native rebuild. W76.
_ap = argparse.ArgumentParser(description=__doc__.split("\n")[1] if __doc__ else None)
_ap.add_argument("--config", required=True,
                 help="Which config states the selected window configuration. Required: "
                      "config.yaml is Channing, config_dn.yaml is deposit-native.")
_ap.add_argument("--results-dir", required=True,
                 help="relative to this script's directory; results_4ct_dn for the "
                      "deposit-native rebuild. Falls back to $NMD_RESULTS_DIR, then "
                      "results_4ct (default: %(default)s)")
_ap.add_argument("--member-seed", type=int, default=None,
                 help="Ensemble member to load, by training seed. Omitted = the legacy "
                      "un-seeded checkpoint. Never silently guesses a member; see "
                      "utils.resolve_checkpoint.")
_args = _ap.parse_args()

RES   = os.path.join(REPO, _args.results_dir)
print(f"[results-dir] {RES}")
# TAG WAS A STRING LITERAL HERE, not derived like every other consumer (2026-07-29). That
# made this the one script that would keep loading the atg500_stop500 checkpoint after a
# sweep selected a different window config -- silently, since the file exists. It now reads
# the one place that names the selection.
# WHICH CONFIG, NOT JUST WHICH TAG (2026-08-04). The note above records making TAG
# config-derived instead of a literal -- and then hardcoded config.yaml to derive it FROM.
# config.yaml is the CHANNING config and its selected: block still reads 500/500, so after the
# deposit-native sweep chose atg1000_stop1000 this script would have loaded the old tag, hunted
# a checkpoint under the old name and built the model with 500/500 windows. Half a fix reads
# exactly like a whole one. The config is now named by the caller, like everywhere else.
config = load_config(_args.config)
TAG   = selected_tag(config)
CKPT  = resolve_checkpoint(RES, TAG, _args.member_seed)
H5    = os.path.join(RES, "nmd_orf_data.h5")
TXSUM = os.path.join(RES, "tx_summary.tsv")
OUT   = os.path.join(RES, "uorf_attention_predictions.tsv")
# WINDOW SIZES COME FROM THE SAME BLOCK AS THE TAG, and must never be stated separately.
# They were the literals `500, 500` while TAG became config-derived, which would have built
# the model with 500/500 windows around a checkpoint trained at whatever the selection said
# -- a shape mismatch if lucky, and silently wrong attention if not. One source, both values.
WS_ATG  = config["selected"]["window_size_atg"]
WS_STOP = config["selected"]["window_size_stop"]
BATCH = 256

# ── Load model ──
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
    # NORMALIZATION STATS COME FROM THE HDF5 BEING READ, not from the literals above.
    #
    # data_prep.py:710-713 writes normalization/orf_feat_{mean,std} into EVERY HDF5,
    # computed from that build's own training split, and utils.py:112-113 -- the path
    # 03_train.py and evaluate.py use -- reads them from there. This script was the only
    # place in the repo that bypassed them, so under --results-dir results_4ct_dn the model
    # was TRAINED with the deposit-native statistics and its inputs normalized here with the
    # PUBLISHED ones. A train/inference mismatch, silent, and it scales every feature.
    #
    # The literals are kept as a fallback for an HDF5 that predates the normalization group,
    # and any disagreement is PRINTED rather than swallowed -- if these numbers move, that is
    # a fact about the universe worth seeing, not a detail to reconcile quietly. W76.
    if "normalization" in f:
        h5_mean = f["normalization/orf_feat_mean"][:].astype(np.float32)
        h5_std = f["normalization/orf_feat_std"][:].astype(np.float32)
        drift = np.abs(h5_mean - NORM_MEAN).max(), np.abs(h5_std - NORM_STD).max()
        print(f"[normalization] from {H5}::/normalization "
              f"(max drift vs the published literals: mean {drift[0]:.4g}, std {drift[1]:.4g})")
        NORM_MEAN, NORM_STD = h5_mean, h5_std
    else:
        print(f"[normalization] !! {H5} carries no /normalization group; falling back to the "
              f"PUBLISHED results_4ct literals baked into this file. If this HDF5 is not the "
              f"published one, its features are being scaled by the wrong constants.")

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

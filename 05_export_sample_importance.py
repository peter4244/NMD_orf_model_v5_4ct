#!/usr/bin/env python3
"""
05_export_sample_importance.py — Export per-sample grad×input vectors for clustering.

Reuses the grad×input computation from 05_interpret_structural.py but saves
the raw per-sample arrays instead of aggregated summaries.

Outputs:
  - sample_importance_{tag}.npz:
      orf_gxi: (n_test, K, n_orf_features) — per-ORF grad×input
      tx_gxi:  (n_test, n_tx_features)     — per-TX grad×input
      labels:  (n_test,)                    — 0/1 NMD labels
      masks:   (n_test, K)                  — valid ORF mask
      isoform_ids: (n_test,)               — isoform identifiers
      orf_feature_names: list of ORF feature names
      tx_feature_names:  list of TX feature names
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import build_model
from utils import NMDDataset, load_config, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--tag", default="atg100_stop500")
    parser.add_argument("--atg-window", type=int, default=None)
    parser.add_argument("--stop-window", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ws_atg = args.atg_window or config["data"]["window_size_atg"]
    ws_stop = args.stop_window or config["data"]["window_size_stop"]
    tag = args.tag
    results_dir = Path("results")
    h5_path = config["data"]["hdf5_path"]

    # Load feature names
    with h5py.File(h5_path, "r") as f:
        orf_feat_names = json.loads(f.attrs["orf_feature_cols"])

    print(f"Exporting per-sample importance for model: {tag}")
    print(f"  ORF features ({len(orf_feat_names)}): {orf_feat_names}")

    # Load model
    ckpt_path = results_dir / f"best_model_{tag}.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_config = {**config["model"],
                    "window_size_atg": ws_atg, "window_size_stop": ws_stop}
    model = build_model(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load test data
    test_ds = NMDDataset(h5_path, ws_atg, ws_stop, split="test_clean")
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"],
                             shuffle=False, num_workers=0)

    # Load isoform IDs using dataset's own indices (robust to split logic changes)
    with h5py.File(h5_path, "r") as f:
        all_ids = np.array([x.decode() if isinstance(x, bytes) else x
                            for x in f["isoform_id"][:]])
    isoform_ids = all_ids[test_ds.indices]

    # Compute grad × input
    print("Computing gradient × input ...")
    all_orf_gxi = []
    all_labels = []
    all_masks = []

    for batch in test_loader:
        atg = batch["atg_windows"].to(device)
        stop = batch["stop_windows"].to(device)
        orf_feat = batch["orf_features"].to(device).requires_grad_(True)
        mask = batch["orf_mask"].to(device)

        cls_logits = model(atg, stop, orf_feat, mask)
        cls_logits.sum().backward()

        all_orf_gxi.append((orf_feat.grad * orf_feat).detach().cpu().numpy())
        all_labels.append(batch["label"].numpy())
        all_masks.append(mask.cpu().numpy())

    orf_gxi = np.concatenate(all_orf_gxi)
    labels = np.concatenate(all_labels)
    masks = np.concatenate(all_masks)

    print(f"  {len(labels):,} test samples, {int(labels.sum()):,} NMD+")
    print(f"  orf_gxi shape: {orf_gxi.shape}")

    # Save
    out_path = results_dir / f"sample_importance_{tag}.npz"
    np.savez_compressed(
        out_path,
        orf_gxi=orf_gxi,
        labels=labels,
        masks=masks,
        isoform_ids=isoform_ids,
        orf_feature_names=orf_feat_names,
    )
    print(f"  -> Saved {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()

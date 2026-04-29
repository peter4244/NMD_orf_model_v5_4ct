#!/usr/bin/env python3
"""
05_interpret_structural.py — Gradient × input attribution for structural features.

Computes feature importance for the 9 ORF-level and 8 transcript-level
structural features using gradient × input. All values are in z-score
normalized space (comparable across features with different scales).

Outputs:
  - structural_importance_orf.tsv: mean |grad × input| per ORF feature, by class
  - structural_importance_tx.tsv: mean |grad × input| per TX feature, by class
  - structural_importance_by_rank.tsv: ORF feature importance stratified by orf_rank and is_ref_cds
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import build_model
from utils import NMDDataset, load_config, set_seed


def compute_grad_x_input(model, dataloader, device, orf_feat_names):
    """v5: Compute gradient × input for ORF structural features only."""
    model.eval()

    all_orf_gxi = []  # (n_samples, K, n_orf_features)
    all_labels = []
    all_masks = []

    for batch in dataloader:
        atg = batch["atg_windows"].to(device)
        stop = batch["stop_windows"].to(device)
        orf_feat = batch["orf_features"].to(device).requires_grad_(True)
        mask = batch["orf_mask"].to(device)

        cls_logits = model(atg, stop, orf_feat, mask)
        cls_logits.sum().backward()

        orf_gxi = (orf_feat.grad * orf_feat).detach().cpu().numpy()

        all_orf_gxi.append(orf_gxi)
        all_labels.append(batch["label"].numpy())
        all_masks.append(mask.cpu().numpy())

    return (np.concatenate(all_orf_gxi),
            np.concatenate(all_labels),
            np.concatenate(all_masks))


def summarize_orf_features(orf_gxi, labels, masks, orf_feat_names, results_dir, tag=""):
    """Mean |grad × input| per ORF feature, stratified by NMD/non-NMD."""
    rows = []
    for label_name, label_val in [("all", None), ("NMD", 1), ("non-NMD", 0)]:
        if label_val is not None:
            idx = labels == label_val
        else:
            idx = np.ones(len(labels), dtype=bool)

        # Pool across all valid ORFs for selected transcripts
        sub_gxi = orf_gxi[idx]      # (n, K, F)
        sub_mask = masks[idx]        # (n, K)

        for f_idx, fname in enumerate(orf_feat_names):
            vals = sub_gxi[:, :, f_idx][sub_mask]  # valid ORFs only
            rows.append({
                "class": label_name,
                "feature": fname,
                "mean_abs_gxi": np.abs(vals).mean(),
                "mean_gxi": vals.mean(),
                "std_gxi": vals.std(),
                "n_orfs": len(vals),
            })

    df = pd.DataFrame(rows)
    suffix = f"_{tag}" if tag else ""
    path = results_dir / f"structural_importance_orf{suffix}.tsv"
    df.to_csv(path, sep="\t", index=False)
    print(f"  -> {path}")

    print("\n  ORF feature importance (NMD class, sorted by |grad × input|):")
    nmd_df = df[df["class"] == "NMD"].sort_values("mean_abs_gxi", ascending=False)
    for _, row in nmd_df.iterrows():
        print(f"    {row['feature']:<25} {row['mean_abs_gxi']:.6f}")

    return df


def summarize_tx_features(tx_gxi, labels, tx_feat_names, results_dir, tag=""):
    """Mean |grad × input| per TX feature, stratified by NMD/non-NMD."""
    rows = []
    for label_name, label_val in [("all", None), ("NMD", 1), ("non-NMD", 0)]:
        if label_val is not None:
            idx = labels == label_val
        else:
            idx = np.ones(len(labels), dtype=bool)

        sub_gxi = tx_gxi[idx]

        for f_idx, fname in enumerate(tx_feat_names):
            vals = sub_gxi[:, f_idx]
            rows.append({
                "class": label_name,
                "feature": fname,
                "mean_abs_gxi": np.abs(vals).mean(),
                "mean_gxi": vals.mean(),
                "std_gxi": vals.std(),
                "n_transcripts": len(vals),
            })

    df = pd.DataFrame(rows)
    suffix = f"_{tag}" if tag else ""
    path = results_dir / f"structural_importance_tx{suffix}.tsv"
    df.to_csv(path, sep="\t", index=False)
    print(f"  -> {path}")

    print("\n  TX feature importance (NMD class, sorted by |grad × input|):")
    nmd_df = df[df["class"] == "NMD"].sort_values("mean_abs_gxi", ascending=False)
    for _, row in nmd_df.iterrows():
        print(f"    {row['feature']:<30} {row['mean_abs_gxi']:.6f}")

    return df


def summarize_by_rank_and_cds(orf_gxi, labels, masks, orf_feat_names,
                               orf_features_raw, results_dir, tag=""):
    """ORF feature importance stratified by orf_rank and is_ref_cds."""
    # is_ref_cds is feature index 7 (after normalization, >0 means ref CDS)
    ref_cds_idx = orf_feat_names.index("is_ref_cds")

    # Only NMD transcripts
    nmd_mask = labels == 1
    gxi = orf_gxi[nmd_mask]
    mask = masks[nmd_mask]
    raw_feat = orf_features_raw[nmd_mask]

    rows = []
    K = gxi.shape[1]

    # By orf_rank
    for rank in range(K):
        valid = mask[:, rank]
        if valid.sum() < 10:
            continue
        for f_idx, fname in enumerate(orf_feat_names):
            vals = gxi[valid, rank, f_idx]
            rows.append({
                "stratification": f"rank_{rank}",
                "feature": fname,
                "mean_abs_gxi": np.abs(vals).mean(),
                "mean_gxi": vals.mean(),
                "n": int(valid.sum()),
            })

    # By is_ref_cds (using raw unnormalized features to get binary indicator)
    for cds_val, cds_name in [(True, "ref_cds"), (False, "non_ref_cds")]:
        if cds_val:
            selector = raw_feat[:, :, ref_cds_idx] > 0
        else:
            selector = raw_feat[:, :, ref_cds_idx] <= 0
        valid = mask & selector
        for f_idx, fname in enumerate(orf_feat_names):
            vals = gxi[:, :, f_idx][valid]
            if len(vals) < 10:
                continue
            rows.append({
                "stratification": cds_name,
                "feature": fname,
                "mean_abs_gxi": np.abs(vals).mean(),
                "mean_gxi": vals.mean(),
                "n": len(vals),
            })

    df = pd.DataFrame(rows)
    suffix = f"_{tag}" if tag else ""
    path = results_dir / f"structural_importance_by_rank{suffix}.tsv"
    df.to_csv(path, sep="\t", index=False)
    print(f"  -> {path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--tag", default="atg20_stop500")
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

    print(f"Structural feature importance for model: {tag}")

    # Load feature names from HDF5
    with h5py.File(h5_path, "r") as f:
        orf_feat_names = json.loads(f.attrs["orf_feature_cols"])

    print(f"  ORF features ({len(orf_feat_names)}): {orf_feat_names}")

    # Load model
    ckpt_path = results_dir / f"best_model_{tag}.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_config = {**config["model"],
                    "window_size_atg": ws_atg, "window_size_stop": ws_stop}
    model = build_model(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Load test data
    test_ds = NMDDataset(h5_path, ws_atg, ws_stop, split="test_clean")
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"],
                             shuffle=False, num_workers=0)

    # Also load raw (unnormalized) ORF features for CDS indicator stratification
    with h5py.File(h5_path, "r") as f:
        splits = np.array([s.decode() if isinstance(s, bytes) else s
                           for s in f["split"][:]])
        test_mask = splits == "test"
        orf_features_raw = f["orf_features"][test_mask].astype(np.float32)

    # Compute grad × input
    print("\nComputing gradient × input ...")
    orf_gxi, labels, masks = compute_grad_x_input(
        model, test_loader, device, orf_feat_names)

    print(f"  {len(labels):,} test samples, {int(labels.sum()):,} NMD+")

    # Summarize
    print("\n=== ORF feature importance ===")
    summarize_orf_features(orf_gxi, labels, masks, orf_feat_names, results_dir, tag)

    print("\n=== Importance by ORF rank and CDS status (NMD only) ===")
    summarize_by_rank_and_cds(orf_gxi, labels, masks, orf_feat_names,
                               orf_features_raw, results_dir, tag)

    print("\n=== Structural importance analysis complete ===")


if __name__ == "__main__":
    main()

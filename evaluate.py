#!/usr/bin/env python3
"""
evaluate.py — Evaluate trained NMD ORF model on holdout set.

Loads best checkpoint, evaluates on test set (chr1,3,5,7),
extracts attention weights, saves predictions and metrics.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from model import NMDOrfModel, build_model
from utils import NMDDataset, compute_metrics, load_config, set_seed


def evaluate(config_path="config.yaml", atg_window=None, stop_window=None):
    config = load_config(config_path)
    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ws_atg = atg_window or config["data"]["window_size_atg"]
    ws_stop = stop_window or config["data"]["window_size_stop"]
    tag = f"atg{ws_atg}_stop{ws_stop}"
    results_dir = Path("results")

    # Load best checkpoint
    ckpt_path = results_dir / f"best_model_{tag}.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_config = {**config["model"],
                    "window_size_atg": ws_atg, "window_size_stop": ws_stop}
    model = build_model(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"  Trained for {ckpt['epoch']} epochs, val AUC = {ckpt['val_auc']:.4f}")

    # Load test data
    h5_path = config["data"]["hdf5_path"]
    test_ds = NMDDataset(h5_path, ws_atg, ws_stop, split="test_clean")
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"],
                             shuffle=False, num_workers=0)

    # Evaluate
    all_labels, all_logits, all_attn = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            atg = batch["atg_windows"].to(device)
            stop = batch["stop_windows"].to(device)
            orf_feat = batch["orf_features"].to(device)
            mask = batch["orf_mask"].to(device)
            cls_logits, attn_weights = model(
                atg, stop, orf_feat, mask, return_attention=True)

            all_labels.extend(batch["label"].numpy())
            all_logits.extend(cls_logits.squeeze(-1).cpu().numpy())
            all_attn.append(attn_weights.cpu().numpy())

    labels = np.array(all_labels)
    logits = np.array(all_logits)
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    attn = np.concatenate(all_attn, axis=0)

    # Metrics
    metrics = compute_metrics(labels, logits)
    metrics["n_test"] = len(labels)
    metrics["n_nmd"] = int(labels.sum())
    metrics["window_size_atg"] = ws_atg
    metrics["window_size_stop"] = ws_stop
    metrics["best_epoch"] = ckpt["epoch"]

    print(f"\nTest set results (n={len(labels):,}):")
    print(f"  AUC:   {metrics['auc']:.4f}")
    print(f"  AUPRC: {metrics['auprc']:.4f}")

    # Load isoform IDs for the test set
    import h5py
    with h5py.File(h5_path, "r") as f:
        all_ids = np.array([s.decode() if isinstance(s, bytes) else s
                            for s in f["isoform_id"][:]])
        all_chrs = np.array([s.decode() if isinstance(s, bytes) else s
                             for s in f["chr"][:]])
    test_ids = all_ids[test_ds.indices]
    test_chrs = all_chrs[test_ds.indices]

    # Save predictions
    pred_df = pd.DataFrame({
        "isoform_id": test_ids,
        "chr": test_chrs,
        "label": labels.astype(int),
        "logit": logits,
        "prob": probs,
    })
    pred_path = results_dir / f"predictions_{tag}.tsv"
    pred_df.to_csv(pred_path, sep="\t", index=False)
    print(f"  -> {pred_path}")

    # Save attention weights
    attn_df_rows = []
    for i in range(len(test_ids)):
        for k in range(attn.shape[1]):
            attn_df_rows.append({
                "isoform_id": test_ids[i],
                "orf_rank": k,
                "attn_weight": attn[i, k],
            })
    attn_df = pd.DataFrame(attn_df_rows)
    attn_path = results_dir / f"attention_weights_{tag}.tsv"
    attn_df.to_csv(attn_path, sep="\t", index=False)
    print(f"  -> {attn_path}")

    # Save metrics
    metrics_path = results_dir / f"metrics_{tag}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  -> {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--atg-window", type=int, default=None,
                        help="Override window_size_atg from config")
    parser.add_argument("--stop-window", type=int, default=None,
                        help="Override window_size_stop from config")
    args = parser.parse_args()
    evaluate(args.config, atg_window=args.atg_window, stop_window=args.stop_window)

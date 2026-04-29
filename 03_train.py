#!/usr/bin/env python3
"""
03_train.py — Training loop for NMD ORF model.

Features:
  - Differential weight decay (higher on CNN, lower on attention/head)
  - Mixed precision training (AMP)
  - Early stopping on validation AUC
  - Overfitting gap monitoring
  - Per-epoch logging to CSV
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from utils import (NMDDataset, compute_metrics, compute_pos_weight,
                   load_config, set_seed)
from model import NMDOrfModel, build_model, count_parameters


def make_optimizer(model, config):
    """Create Adam optimizer with differential weight decay."""
    cnn_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "atg_cnn" in name or "stop_cnn" in name:
            cnn_params.append(param)
        else:
            other_params.append(param)

    return torch.optim.Adam([
        {"params": cnn_params, "weight_decay": config["training"]["weight_decay_cnn"]},
        {"params": other_params, "weight_decay": config["training"]["weight_decay_other"]},
    ], lr=config["training"]["lr"])


def train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0
    all_labels, all_logits = [], []

    for batch in loader:
        atg = batch["atg_windows"].to(device)
        stop = batch["stop_windows"].to(device)
        orf_feat = batch["orf_features"].to(device)
        mask = batch["orf_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        with autocast("cuda", enabled=use_amp):
            cls_logits = model(atg, stop, orf_feat, mask)
            loss = criterion(cls_logits.squeeze(-1), labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(labels)
        all_labels.extend(labels.cpu().numpy())
        all_logits.extend(cls_logits.squeeze(-1).detach().cpu().numpy())

    metrics = compute_metrics(all_labels, all_logits)
    metrics["loss"] = total_loss / len(all_labels)
    return metrics


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0
    all_labels, all_logits = [], []

    for batch in loader:
        atg = batch["atg_windows"].to(device)
        stop = batch["stop_windows"].to(device)
        orf_feat = batch["orf_features"].to(device)
        mask = batch["orf_mask"].to(device)
        labels = batch["label"].to(device)

        with autocast("cuda", enabled=use_amp):
            cls_logits = model(atg, stop, orf_feat, mask)
            loss = criterion(cls_logits.squeeze(-1), labels)

        total_loss += loss.item() * len(labels)
        all_labels.extend(labels.cpu().numpy())
        all_logits.extend(cls_logits.squeeze(-1).cpu().numpy())

    metrics = compute_metrics(all_labels, all_logits)
    metrics["loss"] = total_loss / len(all_labels)
    return metrics


def train(config_path="config.yaml", atg_window=None, stop_window=None):
    config = load_config(config_path)
    set_seed(config["training"]["seed"])

    # Apply CLI overrides for window sizes
    ws_atg = atg_window or config["data"]["window_size_atg"]
    ws_stop = stop_window or config["data"]["window_size_stop"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    h5_path = config["data"]["hdf5_path"]
    print(f"\nLoading data (ATG={ws_atg}, stop={ws_stop}) ...")

    train_ds = NMDDataset(h5_path, ws_atg, ws_stop, split="train")
    val_ds = NMDDataset(h5_path, ws_atg, ws_stop, split="val")

    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"],
                              shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"],
                            shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model_config = {**config["model"],
                    "window_size_atg": ws_atg, "window_size_stop": ws_stop}
    model = build_model(model_config).to(device)

    # Loss
    pos_weight = compute_pos_weight(train_ds).to(device)
    print(f"pos_weight: {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer and scheduler
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=config["training"]["scheduler_patience"],
        factor=config["training"]["scheduler_factor"])

    # Mixed precision
    use_amp = config["training"].get("mixed_precision", False) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    print(f"Mixed precision: {use_amp}")

    # Logging
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    tag = f"atg{ws_atg}_stop{ws_stop}"
    log_path = results_dir / f"training_log_{tag}.csv"
    best_ckpt_path = results_dir / f"best_model_{tag}.pt"

    best_val_auc = 0.0
    patience_counter = 0
    patience = config["training"]["patience"]
    gap_threshold = config["training"].get("overfit_gap_threshold", 0.05)

    with open(log_path, "w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=[
            "epoch", "train_loss", "train_auc", "train_auprc",
            "val_loss", "val_auc", "val_auprc", "auc_gap", "lr", "time_s"])
        writer.writeheader()

        print(f"\nTraining for up to {config['training']['epochs']} epochs ...")
        for epoch in range(1, config["training"]["epochs"] + 1):
            t0 = time.time()

            train_metrics = train_epoch(model, train_loader, criterion,
                                        optimizer, scaler, device, use_amp)
            val_metrics = eval_epoch(model, val_loader, criterion, device, use_amp)

            prev_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_metrics["auc"])
            current_lr = optimizer.param_groups[0]["lr"]
            if current_lr < prev_lr:
                print(f"    -> LR reduced: {prev_lr:.6f} -> {current_lr:.6f}")
            elapsed = time.time() - t0
            auc_gap = train_metrics["auc"] - val_metrics["auc"]

            row = {
                "epoch": epoch,
                "train_loss": f"{train_metrics['loss']:.4f}",
                "train_auc": f"{train_metrics['auc']:.4f}",
                "train_auprc": f"{train_metrics['auprc']:.4f}",
                "val_loss": f"{val_metrics['loss']:.4f}",
                "val_auc": f"{val_metrics['auc']:.4f}",
                "val_auprc": f"{val_metrics['auprc']:.4f}",
                "auc_gap": f"{auc_gap:.4f}",
                "lr": f"{current_lr:.6f}",
                "time_s": f"{elapsed:.1f}",
            }
            writer.writerow(row)
            log_file.flush()

            gap_flag = " *** OVERFIT" if auc_gap > gap_threshold else ""
            print(f"  Epoch {epoch:3d} | train AUC {train_metrics['auc']:.4f} | "
                  f"val AUC {val_metrics['auc']:.4f} | gap {auc_gap:+.4f} | "
                  f"{elapsed:.0f}s{gap_flag}")

            # Checkpointing
            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": best_val_auc,
                    "config": config,
                }, best_ckpt_path)
                print(f"    -> New best val AUC: {best_val_auc:.4f}, saved checkpoint")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n  Early stopping at epoch {epoch} "
                          f"(no improvement for {patience} epochs)")
                    break

    print(f"\nBest val AUC: {best_val_auc:.4f}")
    print(f"Training log: {log_path}")
    print(f"Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--atg-window", type=int, default=None,
                        help="Override window_size_atg from config")
    parser.add_argument("--stop-window", type=int, default=None,
                        help="Override window_size_stop from config")
    args = parser.parse_args()
    train(args.config, atg_window=args.atg_window, stop_window=args.stop_window)

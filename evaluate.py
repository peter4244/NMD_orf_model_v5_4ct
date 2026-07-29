#!/usr/bin/env python3
"""
evaluate.py — Score a trained NMD ORF model on a NAMED split.

Loads one ensemble member's checkpoint, evaluates on the split given by --split (required,
no default), extracts attention weights, saves predictions and metrics.

--split is mandatory and test splits additionally require --final, because this script used
to hardcode split="test_clean": every metrics_{tag}.json in existence is therefore a test-set
score, including the twelve that selected the published window configuration. Model selection
belongs on train/val; --final marks the one evaluation that is allowed to touch chr1,3,5,7.

Outputs are named {tag}[_seed<N>]_{split}, so a selection sweep can never write a file that
looks like a published test-set result.
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
from utils import (NMDDataset, compute_metrics, load_config, member_tag,
                   resolve_checkpoint, set_seed)


FINAL_SPLITS = {"test", "test_clean", "test_all", "test_paralog"}


def evaluate(config_path="config.yaml", atg_window=None, stop_window=None,
             results_dir="results_4ct", member_seed=None, split=None):
    config = load_config(config_path)
    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ws_atg = atg_window or config["data"]["window_size_atg"]
    ws_stop = stop_window or config["data"]["window_size_stop"]
    tag = f"atg{ws_atg}_stop{ws_stop}"
    # See 03_train.py: parameterised so an alternative run cannot overwrite the published
    # predictions/metrics it is meant to be compared with. Default unchanged.
    results_dir = Path(results_dir)

    # EVERY OUTPUT CARRIES ITS MEMBER AND ITS SPLIT (2026-07-29, C9). Without the split in
    # the name, a twelve-config sweep scored on validation writes twelve files named exactly
    # like the published test-set ones -- same defect D-row 4 exists to repair, now
    # indistinguishable on disk. Without the member, five members overwrite each other.
    stem = f"{member_tag(tag, member_seed)}_{split}"

    # Load best checkpoint
    ckpt_path = resolve_checkpoint(results_dir, tag, member_seed)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_config = {**config["model"],
                    "window_size_atg": ws_atg, "window_size_stop": ws_stop}
    model = build_model(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"  Trained for {ckpt['epoch']} epochs, val AUC = {ckpt['val_auc']:.4f}")

    # Load the requested split. NOT hardcoded to test_clean any more (2026-07-29) -- that
    # hardcode is why every metrics_{tag}.json in existence is test-only, and why the window
    # sweep that chose the published config was scored on the held-out set.
    h5_path = config["data"]["hdf5_path"]
    test_ds = NMDDataset(h5_path, ws_atg, ws_stop, split=split)
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
    # THE SPLIT IS RECORDED IN THE FILE, not implied by the filename (2026-07-29). The
    # published metrics_{tag}.json files record `n_test` and no split name, which is what let
    # twelve test-set sweep scores be read later as if the choice had been made on validation
    # data. A number whose population is not written down next to it is the dominant defect
    # class in this manuscript.
    metrics["split"] = split
    metrics["n_eval"] = len(labels)
    # `n_test` is emitted ONLY for genuine test splits. Legacy consumers keep working on test
    # runs and raise KeyError on a val run -- which is correct: a validation score must not be
    # silently readable as a test score.
    if split in FINAL_SPLITS:
        metrics["n_test"] = len(labels)
    metrics["n_nmd"] = int(labels.sum())
    metrics["window_size_atg"] = ws_atg
    metrics["window_size_stop"] = ws_stop
    metrics["best_epoch"] = ckpt["epoch"]
    metrics["member_seed"] = member_seed

    print(f"\n{split} results (n={len(labels):,}):")
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
    pred_path = results_dir / f"predictions_{stem}.tsv"
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
    attn_path = results_dir / f"attention_weights_{stem}.tsv"
    attn_df.to_csv(attn_path, sep="\t", index=False)
    print(f"  -> {attn_path}")

    # Save metrics
    metrics_path = results_dir / f"metrics_{stem}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  -> {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--results-dir", default="results_4ct",
                        help="where the checkpoint is read and outputs written")
    parser.add_argument("--atg-window", type=int, default=None,
                        help="Override window_size_atg from config")
    parser.add_argument("--stop-window", type=int, default=None,
                        help="Override window_size_stop from config")
    parser.add_argument("--member-seed", type=int, default=None, help="Ensemble member to load, by training seed. Omitted = the legacy un-seeded checkpoint. Never silently guesses a member; see utils.resolve_checkpoint.")
    # NO DEFAULT, DELIBERATELY. `split` used to be the literal "test_clean" inside the
    # function, so every score this script has ever produced is a test-set score -- including
    # the twelve that chose the published window configuration. A default here, of any value,
    # would let that happen again without anyone typing the word "test".
    parser.add_argument("--split", required=True,
                        choices=["train", "val", "val_clean", "val_all",
                                 "test", "test_clean", "test_all", "test_paralog", "all"],
                        help="Which split to score. REQUIRED. Model selection must use "
                             "train/val splits; test splits additionally require --final.")
    parser.add_argument("--final", action="store_true",
                        help="Affirm that this is a final, pre-registered evaluation. Required "
                             "before any test split can be scored.")
    args = parser.parse_args()

    # THE TEST-SET GATE. Selecting a configuration on the test set is the leak D-row 4 exists
    # to repair; it happened because scoring the test set was the path of least resistance.
    # It now costs an extra, explicitly-named flag, so it cannot be done inattentively.
    if args.split in FINAL_SPLITS and not args.final:
        parser.error(
            f"--split {args.split} is a TEST split. Scoring it is a final evaluation and "
            f"requires --final.\nIf you are selecting a window configuration, tuning, or "
            f"comparing runs, use --split val_clean (or train). The published configuration "
            f"was chosen on twelve test-set scores; that is the defect this gate prevents.")
    if args.final and args.split not in FINAL_SPLITS:
        parser.error(f"--final given with --split {args.split}, which is not a test split. "
                     f"--final asserts a final evaluation; drop it, or name a test split.")

    evaluate(args.config, atg_window=args.atg_window, stop_window=args.stop_window,
             results_dir=args.results_dir, member_seed=args.member_seed, split=args.split)

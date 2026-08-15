#!/usr/bin/env python3
"""
evaluate.py — Score a trained NMD ORF model on a NAMED split.

Loads one ensemble member's checkpoint, evaluates on the split given by --split (required,
no default), extracts attention weights, saves predictions and metrics.

--split is mandatory and test splits additionally require --final, because this script used
to hardcode split="test_clean": every metrics_{tag}.json in existence is therefore a test-set
score, including the twelve that selected the published window configuration. Model selection
belongs on train/val.

TWO splits reach chr1/3/5/7 and each needs its own affirmation (the second added 2026-07-30):

  --final        a test split: one shot, pre-registered, a genuine held-out estimate.
  --full-cohort  --split all: EVERY isoform, train and test pooled. Legitimate for
                 interpretation (attention, DeepSHAP, branch decomposition), never for
                 performance. `all` previously required neither flag and could not even be
                 marked final, so the held-out chromosomes could be scored without anyone
                 typing "test".

The affirmation propagates rather than staying at the command line:
  * metrics JSON carries `evaluation_class` (development | final_test | full_cohort)
  * a full-cohort run writes NO `auc`/`auprc` key -- they become `auc_mixed_in_sample` /
    `auprc_mixed_in_sample`, so a consumer expecting `auc` raises instead of quoting an
    in-sample number, the same way `n_test` already protects validation runs
  * predictions_*.tsv carries a per-row `split` column, so a pooled file is self-describing
    and the legitimate held-out metric is recoverable by filtering

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

# SPLITS THAT POOL TRAINING AND HELD-OUT DATA (2026-07-30).
#
# `all` maps to np.ones(...) in NMDDataset -- EVERY isoform, chr1/3/5/7 included. It was not in
# FINAL_SPLITS, so it scored the held-out chromosomes with no --final, and the second gate below
# made it impossible to even mark such a run as final. The --final docstring claimed "--final
# marks the one evaluation that is allowed to touch chr1,3,5,7"; that was false while `all`
# existed ungated.
#
# `all` is legitimate and stays: the interpretation analyses run over the full cohort by design
# (attention, DeepSHAP, branch decomposition). What is NOT legitimate is a ranking metric computed
# over it, because train and test are pooled -- so this asks for its own affirmation, and the
# metrics it writes are named so they cannot be read as generalization performance.
FULL_COHORT_SPLITS = {"all"}

# Written into every metrics JSON so the population is carried BY THE FILE, not inferred from the
# filename. `development` = train/val, safe for selection. `final_test` = held-out, one shot.
# `full_cohort` = train and test pooled; interpretation only, never a performance number.
EVAL_CLASS = {"final_test": FINAL_SPLITS, "full_cohort": FULL_COHORT_SPLITS}


def eval_class_of(split):
    for cls, members in EVAL_CLASS.items():
        if split in members:
            return cls
    return "development"


def enforce_split_gate(parser, args):
    """The ONE definition of the --final / --full-cohort gates. Both entry points call this.

    EXTRACTED 2026-07-30 after review. ensemble_evaluate.py imported the CONSTANTS from here and
    then restated all four enforcement conditions as its own parser.error calls -- shared data,
    duplicated logic -- while its docstring claimed the gates were "imported, not reimplemented".
    That is the dangerous kind of false: the next person edits the gate here, reads that docstring,
    and believes the ensemble path inherited the change. The four conditions were still faithful
    copies when reviewed, so nothing had diverged yet; the defect was structural, and the fix is to
    make divergence impossible rather than to check for it.

    It also makes the gate TESTABLE. A planted-defect test against duplicated conditions confirms
    that each copy fires and has no power to detect that they are copies. With one writer, a test
    against this function covers every caller.

    D31 and claim 5.6.4 rest on the test set being touched once, deliberately. That is what this
    protects.
    """
    if args.split in FINAL_SPLITS and not args.final:
        parser.error(
            f"--split {args.split} is a TEST split. Scoring it is a final evaluation and "
            f"requires --final.\nIf you are selecting a window configuration, tuning, or "
            f"comparing runs, use --split val_clean (or train). The published configuration "
            f"was chosen on twelve test-set scores; that is the defect this gate prevents.")
    if args.final and args.split not in FINAL_SPLITS:
        parser.error(f"--final given with --split {args.split}, which is not a test split. "
                     f"--final asserts a final evaluation; drop it, or name a test split.")
    if args.split in FULL_COHORT_SPLITS and not getattr(args, "full_cohort", False):
        parser.error(
            f"--split {args.split} POOLS training and held-out data (every isoform, chr1/3/5/7 "
            f"included).\nAny ranking metric over it is in-sample and is not model performance. "
            f"If you want the full cohort for an interpretation analysis -- attention, DeepSHAP, "
            f"branch decomposition -- pass --full-cohort to say so.\nIf you want a performance "
            f"number, use --split val_clean, or --split test_clean --final.")
    if getattr(args, "full_cohort", False) and args.split not in FULL_COHORT_SPLITS:
        parser.error(f"--full-cohort given with --split {args.split}, which is not a pooled "
                     f"split. Drop it.")


# results_dir HAS NO DEFAULT. It was "results_4ct" -- the PUBLISHED run -- so any caller
# that omitted it trained/scored against the old universe and exited 0. The CLI below always
# passes it, so this default was reachable only by a direct import, which is exactly the
# caller with no wrapper to correct it. None + a loud check beats a silent wrong tree.
def evaluate(config_path, atg_window=None, stop_window=None,
             results_dir=None, member_seed=None, split=None):
    if not results_dir:
        raise ValueError("evaluate() needs an explicit results_dir; there is no default tree.")
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
    # THE SCOPE OF THE METRIC IS PART OF THE METRIC (2026-07-30). For a full-cohort split the
    # score is computed over pooled train+test data, so it is not a generalization estimate and
    # must not be readable as one. `auc`/`auprc` are therefore RENAMED rather than annotated:
    # a consumer that expects `auc` raises KeyError instead of quoting an in-sample number --
    # the same mechanism `n_test` already uses for validation runs.
    ev_class = eval_class_of(split)
    if ev_class == "full_cohort":
        metrics["auc_mixed_in_sample"] = metrics.pop("auc")
        metrics["auprc_mixed_in_sample"] = metrics.pop("auprc")
        metrics["metric_scope"] = (
            "POOLED train+val+test. In-sample for the training rows. NOT a generalization "
            "estimate and NOT comparable to a test-split AUC. For a legitimate held-out "
            "number, filter predictions_*.tsv to split == 'test' or re-run with "
            "--split test_clean --final.")
    # THE SPLIT IS RECORDED IN THE FILE, not implied by the filename (2026-07-29). The
    # published metrics_{tag}.json files record `n_test` and no split name, which is what let
    # twelve test-set sweep scores be read later as if the choice had been made on validation
    # data. A number whose population is not written down next to it is the dominant defect
    # class in this manuscript.
    metrics["split"] = split
    metrics["evaluation_class"] = ev_class
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

    print(f"\n{split} results (n={len(labels):,}), evaluation_class={ev_class}:")
    if ev_class == "full_cohort":
        print(f"  AUC (POOLED, in-sample):   {metrics['auc_mixed_in_sample']:.4f}")
        print(f"  AUPRC (POOLED, in-sample): {metrics['auprc_mixed_in_sample']:.4f}")
        print("  ^ train and test are pooled here. NOT a generalization estimate; do not quote.")
    else:
        print(f"  AUC:   {metrics['auc']:.4f}")
        print(f"  AUPRC: {metrics['auprc']:.4f}")

    # Load isoform IDs for the test set
    import h5py
    with h5py.File(h5_path, "r") as f:
        all_ids = np.array([s.decode() if isinstance(s, bytes) else s
                            for s in f["isoform_id"][:]])
        all_chrs = np.array([s.decode() if isinstance(s, bytes) else s
                             for s in f["chr"][:]])
        # PER-ROW SPLIT LABEL (2026-07-30). A pooled `--split all` predictions file was previously
        # indistinguishable, row by row, from a test-set one, so anyone recomputing an AUC from it
        # got an in-sample number with nothing on the row saying so. Carrying the HDF5's own split
        # label makes the file self-describing AND makes the legitimate held-out metric recoverable
        # by filtering, instead of requiring a second evaluation run.
        all_splits = np.array([s.decode() if isinstance(s, bytes) else s
                               for s in f["split"][:]])
    test_ids = all_ids[test_ds.indices]
    test_chrs = all_chrs[test_ds.indices]
    test_splits = all_splits[test_ds.indices]

    # Save predictions
    pred_df = pd.DataFrame({
        "isoform_id": test_ids,
        "split": test_splits,
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
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-dir", required=True,
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
    parser.add_argument("--full-cohort", action="store_true",
                        help="Affirm that this run pools training and held-out data for an "
                             "INTERPRETATION analysis, not for performance. Required for "
                             "--split all, and refused for anything else. The metrics file will "
                             "carry no `auc`/`auprc` key, by design.")
    args = parser.parse_args()

    # ONE gate, one writer, called by every scoring entry point. See enforce_split_gate.
    enforce_split_gate(parser, args)

    evaluate(args.config, atg_window=args.atg_window, stop_window=args.stop_window,
             results_dir=args.results_dir, member_seed=args.member_seed, split=args.split)

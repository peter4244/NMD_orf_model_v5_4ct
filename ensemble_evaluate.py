#!/usr/bin/env python3
"""ensemble_evaluate.py — the five-member ensemble D30 specifies, and the member spread beside it.

WHY THIS FILE EXISTS. Until 2026-07-30 nothing in this repository computed an ensemble anything:
METHODS.md marked the section [PLANNED -- NOT IMPLEMENTED] and that was accurate.
`slurm_train_ensemble_dn.sh` trains five members and evaluates each SEPARATELY, so the deliverable
D30 actually names -- one number from five models -- had no producer.

D30, restated so the code can be checked against it:

  * the ensemble prediction is the mean **LOGIT** across independently seeded members;
  * weights are **never** averaged -- five models are run and their outputs combined, which is a
    different estimator and the only one that is defined here;
  * headline performance is reported **alongside** the single-model mean +/- sd across members, so
    the ensemble's gain over one model is visible rather than implied.

WHY MEAN LOGIT AND NOT MEAN PROBABILITY. They are different estimators and the choice changes the
number. Averaging in logit space is the geometric mean of the odds; averaging probabilities is the
arithmetic mean. D30 says logit, so that is what this computes -- and it records
`aggregation: "mean_logit"` in the metrics file so a reader never has to infer which was used. AUC
is invariant to the monotone sigmoid, so it is unaffected by the choice; AUPRC and any threshold
metric are NOT.

THE SPLIT GATES ARE IMPORTED, NOT REIMPLEMENTED. evaluate.py's --final and --full-cohort gates
exist because the published configuration was selected on twelve test-set scores. A second scoring
entry point with its own copy of that logic is a second place for it to rot, and the copy that
drifts is the one nobody is looking at. This module imports the gate and the vocabulary from
evaluate.py so there is exactly one definition.

    python3 ensemble_evaluate.py --atg-window 2000 --stop-window 100 \
        --results-dir results_4ct_sweep --seeds 100,200,300,400,500 --split val_clean
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from evaluate import FINAL_SPLITS, FULL_COHORT_SPLITS, eval_class_of
from model import build_model
from utils import (NMDDataset, compute_metrics, load_config, member_tag,
                   resolve_checkpoint, set_seed)


def score_member(h5_path, cfg, ws_atg, ws_stop, split, ckpt_path, device, batch):
    """Logits for one member over `split`, in dataset order."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model({**cfg["model"],
                         "window_size_atg": ws_atg, "window_size_stop": ws_stop}).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = NMDDataset(h5_path, ws_atg, ws_stop, split=split)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    logits, labels = [], []
    with torch.no_grad():
        for b in loader:
            out = model(b["atg_windows"].to(device), b["stop_windows"].to(device),
                        b["orf_features"].to(device), b["orf_mask"].to(device))
            logits.extend(out.squeeze(-1).cpu().numpy())
            labels.extend(b["label"].numpy())
    return np.asarray(logits, float), np.asarray(labels, float), ds.indices, int(ckpt["epoch"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--results-dir", default="results_4ct")
    ap.add_argument("--atg-window", type=int, required=True)
    ap.add_argument("--stop-window", type=int, required=True)
    ap.add_argument("--seeds", default="100,200,300,400,500",
                    help="comma-separated member seeds; every one must have a checkpoint")
    ap.add_argument("--split", required=True,
                    choices=["train", "val", "val_clean", "val_all",
                             "test", "test_clean", "test_all", "test_paralog", "all"])
    ap.add_argument("--final", action="store_true",
                    help="affirm a final, pre-registered test evaluation")
    ap.add_argument("--full-cohort", action="store_true",
                    help="affirm a pooled train+test interpretation run (--split all)")
    a = ap.parse_args()

    # SAME GATES AS evaluate.py, imported rather than restated.
    if a.split in FINAL_SPLITS and not a.final:
        ap.error(f"--split {a.split} is a TEST split and requires --final. The published "
                 f"configuration was chosen on twelve test-set scores; that is the defect this "
                 f"gate prevents. Use --split val_clean for anything developmental.")
    if a.final and a.split not in FINAL_SPLITS:
        ap.error(f"--final given with --split {a.split}, which is not a test split.")
    if a.split in FULL_COHORT_SPLITS and not a.full_cohort:
        ap.error(f"--split {a.split} POOLS training and held-out data. Pass --full-cohort to "
                 f"affirm an interpretation run; it is not a performance number.")
    if a.full_cohort and a.split not in FULL_COHORT_SPLITS:
        ap.error(f"--full-cohort given with --split {a.split}, which is not a pooled split.")

    cfg = load_config(a.config)
    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rd = Path(a.results_dir)
    tag = f"atg{a.atg_window}_stop{a.stop_window}"
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    if len(seeds) < 2:
        ap.error("an ensemble needs at least 2 members; got " + str(seeds))

    # Resolve EVERY member before scoring any: a run that dies halfway leaves a partial ensemble
    # whose member count is not what the filename says.
    paths = [resolve_checkpoint(rd, tag, s) for s in seeds]
    print(f"ensemble: {tag}, {len(seeds)} members {seeds}, split={a.split}")

    h5 = cfg["data"]["hdf5_path"]
    per_member, labels_ref, idx_ref, epochs = [], None, None, []
    for s, p in zip(seeds, paths):
        lg, lb, idx, ep = score_member(h5, cfg, a.atg_window, a.stop_window, a.split,
                                       p, device, cfg["training"]["batch_size"])
        # Every member must have scored the SAME transcripts in the SAME order, or the mean is
        # taken across misaligned rows and still produces a plausible number.
        if labels_ref is None:
            labels_ref, idx_ref = lb, idx
        else:
            if not np.array_equal(idx, idx_ref):
                raise ValueError(f"member seed {s} scored a different transcript set")
            if not np.array_equal(lb, labels_ref):
                raise ValueError(f"member seed {s} disagrees on labels")
        per_member.append(lg)
        epochs.append(ep)
        m = compute_metrics(lb, lg)
        print(f"  seed {s:<4} epoch {ep:<3} AUC {m['auc']:.5f}  AUPRC {m['auprc']:.5f}")

    L = np.vstack(per_member)                    # (n_members, n_transcripts)
    ens_logit = L.mean(axis=0)                   # D30: mean LOGIT, never averaged weights
    ens = compute_metrics(labels_ref, ens_logit)
    mem_auc = np.array([compute_metrics(labels_ref, l)["auc"] for l in L])
    mem_ap = np.array([compute_metrics(labels_ref, l)["auprc"] for l in L])

    ev_class = eval_class_of(a.split)
    out = {
        "tag": tag, "split": a.split, "evaluation_class": ev_class,
        "aggregation": "mean_logit",
        "n_members": len(seeds), "member_seeds": seeds, "member_best_epochs": epochs,
        "n_eval": int(labels_ref.size), "n_nmd": int(labels_ref.sum()),
        "window_size_atg": a.atg_window, "window_size_stop": a.stop_window,
        "member_auc_mean": float(mem_auc.mean()), "member_auc_sd": float(mem_auc.std(ddof=1)),
        "member_auprc_mean": float(mem_ap.mean()), "member_auprc_sd": float(mem_ap.std(ddof=1)),
    }
    # A pooled split is not a performance estimate; name its metrics so they cannot be read as one.
    if ev_class == "full_cohort":
        out["ensemble_auc_mixed_in_sample"] = ens["auc"]
        out["ensemble_auprc_mixed_in_sample"] = ens["auprc"]
        out["metric_scope"] = ("POOLED train+val+test; in-sample for the training rows. NOT a "
                               "generalization estimate.")
    else:
        out["ensemble_auc"] = ens["auc"]
        out["ensemble_auprc"] = ens["auprc"]
    if a.split in FINAL_SPLITS:
        out["n_test"] = int(labels_ref.size)

    print(f"\n  ENSEMBLE (mean logit, n={len(seeds)}): "
          f"AUC {ens['auc']:.5f}  AUPRC {ens['auprc']:.5f}")
    print(f"  single model across members: AUC {mem_auc.mean():.5f} +/- {mem_auc.std(ddof=1):.5f}"
          f"   AUPRC {mem_ap.mean():.5f} +/- {mem_ap.std(ddof=1):.5f}")
    print(f"  ensemble gain over the mean member: AUC {ens['auc']-mem_auc.mean():+.5f}  "
          f"AUPRC {ens['auprc']-mem_ap.mean():+.5f}")

    stem = f"{tag}_ens{len(seeds)}_{a.split}"
    mp = rd / f"ensemble_metrics_{stem}.json"
    mp.write_text(json.dumps(out, indent=2))
    print(f"  -> {mp}")

    import h5py
    with h5py.File(h5, "r") as f:
        ids = np.array([s.decode() if isinstance(s, bytes) else s for s in f["isoform_id"][:]])
        chrs = np.array([s.decode() if isinstance(s, bytes) else s for s in f["chr"][:]])
        sp = np.array([s.decode() if isinstance(s, bytes) else s for s in f["split"][:]])
    df = pd.DataFrame({"isoform_id": ids[idx_ref], "split": sp[idx_ref], "chr": chrs[idx_ref],
                       "label": labels_ref.astype(int)})
    for s, l in zip(seeds, L):
        df[f"logit_seed{s}"] = l
    df["logit_ensemble"] = ens_logit
    df["prob_ensemble"] = torch.sigmoid(torch.tensor(ens_logit)).numpy()
    pp = rd / f"ensemble_predictions_{stem}.tsv"
    df.to_csv(pp, sep="\t", index=False)
    print(f"  -> {pp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

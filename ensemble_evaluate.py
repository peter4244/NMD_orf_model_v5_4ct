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

WHY MEAN LOGIT AND NOT MEAN PROBABILITY, corrected 2026-07-30 after review. An earlier version of
this note said "AUC is invariant to the monotone sigmoid so it is unaffected by the choice; AUPRC
and any threshold metric are NOT." That conflated two different things and was wrong about AUPRC.

  * THE SIGMOID IS INERT for both metrics. compute_metrics applies sigmoid then calls
    roc_auc_score and average_precision_score; both are RANK-based, and a monotone transform
    cannot change a ranking. Verified numerically: AP identical to ten decimals with and without.
    What genuinely is not invariant is anything at a FIXED threshold (accuracy, F1 at 0.5) and
    every calibration measure (Brier, log-loss) -- none of which this reports.
  * THE AGGREGATION IS NOT INERT, and it moves BOTH metrics. mean(sigmoid(x)) != sigmoid(mean(x)),
    so mean-logit and mean-probability produce DIFFERENT RANKINGS. Measured on synthetic data with
    realistic class imbalance: 3,986 of 4,000 ranks differ, AUC by +0.0047 and AUPRC by +0.0209 --
    the AUPRC difference alone being larger than this sweep's exclusion threshold.

So the choice matters and it is pre-registered rather than convenient: D30 names mean LOGIT, and
swapping a pre-registered choice after seeing an 80-cell grid is exactly what D34/W114 exist to
catch. Mean logit is primary. Mean probability is computed ALONGSIDE as a declared sensitivity
check -- both are recorded, `aggregation: "mean_logit"` names the primary, and a reader never has
to infer which produced the headline. Expectation, stated before the first run: they should agree
closely if the members are similarly calibrated, and any divergence should localise to the
low-score tail, which is where AUPRC lives under imbalance.

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

from evaluate import FINAL_SPLITS, FULL_COHORT_SPLITS, enforce_split_gate, eval_class_of
from model import build_model
from utils import (NMDDataset, compute_metrics, load_config, member_tag,
                   resolve_checkpoint, set_seed)


def score_member(loader, cfg, ws_atg, ws_stop, ckpt_path, device):
    """Logits for one member over an ALREADY-LOADED dataset, in dataset order.

    THE DATASET IS A PARAMETER, NOT BUILT HERE (fixed 2026-07-30, before this ever ran). The first
    version constructed a fresh NMDDataset per member. Every member scores the SAME split with the
    SAME window sizes, so that was five identical loads: NMDDataset materialises the whole split in
    RAM as float32, ~1.65 GB for atg2000_stop100 on val_clean, so a five-member run churned ~8 GB
    to read one dataset. It also removed the possibility of a member scoring a different transcript
    set, which the alignment guard below then had to detect after the fact. Loading once makes that
    unrepresentable rather than merely checked.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model({**cfg["model"],
                         "window_size_atg": ws_atg, "window_size_stop": ws_stop}).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logits, labels, iter_order = [], [], []
    n_seen = 0
    with torch.no_grad():
        for b in loader:
            out = model(b["atg_windows"].to(device), b["stop_windows"].to(device),
                        b["orf_features"].to(device), b["orf_mask"].to(device))
            logits.extend(out.squeeze(-1).cpu().numpy())
            labels.extend(b["label"].numpy())
            # Record the position each row occupied in THIS pass. With shuffle=False this is
            # 0,1,2,...; if a future refactor turns shuffling on, or swaps in a sampler, this
            # diverges between members and the guard below catches it. Recording what was
            # ITERATED is the only thing that can -- the index SET is unchanged by a reordering.
            iter_order.extend(range(n_seen, n_seen + len(b["label"])))
            n_seen += len(b["label"])
    # Return a COPY of the indices, and the order they were actually iterated in.
    #
    # Two previous versions returned `ds.indices` / `loader.dataset.indices` -- the SAME OBJECT the
    # caller already held -- so the check was `array_equal(x, x)` and could never fire. A copy is
    # the minimum that makes the comparison mean anything; `iter_order` is what makes it detect the
    # failure it exists for, since a reordering leaves the index SET identical.
    return (np.asarray(logits, float), np.asarray(labels, float),
            np.array(loader.dataset.indices, copy=True), np.asarray(iter_order), int(ckpt["epoch"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
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

    # THE gate, called not copied. The previous version restated all four conditions here while
    # the docstring claimed they were imported -- shared constants, duplicated logic, and a
    # duplicate that stops tracking its twin is the failure D31 and claim 5.6.4 cannot afford.
    enforce_split_gate(ap, a)

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

    # DISTINCT FILES, CHECKED BEFORE SCORING. resolve_checkpoint refuses to guess a member, but it
    # cannot know that two seeds resolved to the same path.
    if len({str(p) for p in paths}) != len(paths):
        raise ValueError(f"member checkpoints are not distinct files: {[str(p) for p in paths]}")

    h5 = cfg["data"]["hdf5_path"]
    ds = NMDDataset(h5, a.atg_window, a.stop_window, split=a.split)   # ONCE, reused by every member
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0)
    idx_ref = ds.indices

    # ALIGNMENT: the structural fix is what protects this, not the checks (review, 2026-07-30).
    # ds and loader are built ONCE above, shuffle=False, num_workers=0, drop_last unset; every
    # member iterates the same loader object and score_member cannot rebuild it. Misalignment is
    # therefore unrepresentable rather than merely detected.
    #
    # The checks below are belt-and-braces against a future refactor reintroducing per-member
    # loaders -- and one of them replaces a guard that could not fire. Comparing LABELS was nearly
    # powerless: labels are 0/1, so ANY permutation within a class leaves the vector identical, and
    # reordering is exactly the failure it was supposed to catch. Verified: a within-class shuffle
    # of 4,000 rows leaves array_equal True while the index vector differs. The INDICES are the
    # only per-row identity available, and until now they were captured once and used solely to
    # build the output frame -- a guard sitting unused.
    per_member, labels_ref, epochs, order_ref = [], None, [], None
    for s, p in zip(seeds, paths):
        lg, lb, midx, morder, ep = score_member(loader, cfg, a.atg_window, a.stop_window, p, device)
        if order_ref is None:
            order_ref = morder
        elif not np.array_equal(morder, order_ref):
            raise ValueError(
                f"member seed {s} iterated the dataset in a DIFFERENT ORDER. The index set can be "
                f"identical while the order differs, so the mean would be taken across misaligned "
                f"rows and would still produce a plausible AUC.")
        if midx is idx_ref:
            raise AssertionError(
                "member indices are the SAME OBJECT as the reference -- array_equal(x, x) is "
                "always True and this guard cannot fire. See test_guards.py.")
        if not np.array_equal(midx, idx_ref):
            raise ValueError(
                f"member seed {s} iterated a DIFFERENT transcript set than the first member. "
                f"The mean would be taken across misaligned rows and would still produce a "
                f"plausible AUC.")
        if labels_ref is None:
            labels_ref = lb
        elif not np.array_equal(lb, labels_ref):
            raise ValueError(f"member seed {s} disagrees on labels")
        per_member.append(lg)
        epochs.append(ep)
        m = compute_metrics(lb, lg)
        print(f"  seed {s:<4} epoch {ep:<3} AUC {m['auc']:.5f}  AUPRC {m['auprc']:.5f}")

    L = np.vstack(per_member)                    # (n_members, n_transcripts)

    # THE MEMBERS MUST ACTUALLY DIFFER -- this repo has already shipped the failure this catches.
    # utils.member_tag records it: 03_train.py once took its seed from config only and wrote
    # best_model_{tag}.pt with no seed slot, so five array tasks torch.save'd to ONE filename and
    # "the five members of the ensemble are one checkpoint written five times". Averaging five
    # identical models yields a believable ensemble AUC and a member sd of exactly 0.0, which reads
    # as a stability result and is a filename collision. Distinct paths do not prove distinct
    # weights, so compare the OUTPUTS.
    dup = [(seeds[i], seeds[j])
           for i in range(len(L)) for j in range(i + 1, len(L))
           if np.array_equal(L[i], L[j])]
    if dup:
        raise ValueError(
            f"members produced IDENTICAL logits for seed pair(s) {dup}. These are not independent "
            f"members -- either the checkpoints are copies, or training ignored --seed. An "
            f"'ensemble' of one model repeated reports member sd = 0 and looks like stability.")
    off = [abs(float(np.corrcoef(L[i], L[j])[0, 1]))
           for i in range(len(L)) for j in range(i + 1, len(L))]
    print(f"  member-to-member logit correlation: min {min(off):.4f} max {max(off):.4f} "
          f"(1.0000 across the board would mean the members are not independent)")
    ens_logit = L.mean(axis=0)                   # D30: mean LOGIT, never averaged weights
    ens = compute_metrics(labels_ref, ens_logit)

    # DECLARED SENSITIVITY CHECK, not an alternative headline. mean(sigmoid) != sigmoid(mean), so
    # this ranks transcripts differently and moves both metrics; on synthetic data with this class
    # balance it shifted AUPRC by 0.021, larger than the sweep's own exclusion threshold. Mean
    # logit stays primary because D30 pre-registered it AND because it is the model's native
    # scale -- cls_head is a bare Linear and the loss is BCEWithLogitsLoss, so probabilities only
    # exist after a sigmoid this pipeline adds. Reporting both means the choice is measured rather
    # than argued, and neither can be quietly substituted for the other.
    ens_prob = (1.0 / (1.0 + np.exp(-L))).mean(axis=0)
    ens_alt = compute_metrics(labels_ref, ens_prob)
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
    # Key names follow the SAME rule as the headline metrics: on a pooled split they must not be
    # readable as performance. Previously this block wrote a bare "auc" even under --full-cohort,
    # directly beneath a comment saying pooled metrics are named so they cannot be read as one.
    _sfx = "_mixed_in_sample" if ev_class == "full_cohort" else ""
    out["sensitivity_mean_probability"] = {
        f"auc{_sfx}": ens_alt["auc"], f"auprc{_sfx}": ens_alt["auprc"],
        "auc_minus_primary": ens_alt["auc"] - ens["auc"],
        "auprc_minus_primary": ens_alt["auprc"] - ens["auprc"],
        "note": ("mean(sigmoid(logit)) instead of sigmoid(mean(logit)). NOT the headline; D30 "
                 "pre-registers mean logit, which is also the model's native output scale. "
                 "Recorded so the aggregation choice is measured rather than assumed."),
    }
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
    print(f"  [sensitivity] mean-PROBABILITY aggregation: AUC {ens_alt['auc']:.5f} "
          f"({ens_alt['auc']-ens['auc']:+.5f})  AUPRC {ens_alt['auprc']:.5f} "
          f"({ens_alt['auprc']-ens['auprc']:+.5f})   -- not the headline")

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

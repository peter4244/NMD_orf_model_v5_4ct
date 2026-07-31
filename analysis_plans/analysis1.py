#!/usr/bin/env python3
"""
analysis1.py — held-out performance of the NMD sequence model.

Implements ANALYSIS1_PLAN_2026-07-30.md. The seven functions below are the plan's seven
steps, in order and under the same names.

EVERY CHECK PRINTS THE VALUE IT MEASURED. A check that only printed OK/FAIL would make this
log a claim about the data; printing the measured value makes the log the evidence. Read the
output next to the plan rather than reading this file.

Run:
    ~/miniforge3/envs/nmd_model_local/bin/python analysis1.py

The default python on this machine has a torch/numpy ABI mismatch that breaks tensor.numpy();
nmd_model_local does not.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

# --- the model architecture is a given artifact, not something this analysis defines ---
MODEL_REPO = Path.home() / "claude_projects" / "NMD_orf_model_v5_4ct"
sys.path.insert(0, str(MODEL_REPO))
from model import build_model  # noqa: E402

# ----------------------------------------------------------------------------------------
# Constants. These are the plan's, verbatim.
# ----------------------------------------------------------------------------------------
H5        = MODEL_REPO / "results_4ct_dn" / "nmd_orf_data.h5"
CKPT_DIR  = MODEL_REPO / "results_4ct_sweep"
CONFIG    = MODEL_REPO / "config_dn.yaml"
OUT_DIR   = MODEL_REPO / "results_interp_all"

TAGS      = ["atg2000_stop2000", "atg1000_stop1000"]
SEEDS     = [100, 200, 300, 400, 500]
B         = 2000        # bootstrap resamples
RSEED     = 20260730    # bootstrap RNG seed, fixed
SPLIT     = "test"      # test_clean is `split == "test"`; it excludes test_paralog
BATCH     = 256         # isoforms per forward pass; memory only, does not affect results

_step = [""]


def log(msg=""):
    print(f"{_step[0]}{msg}", flush=True)


def step(n, title):
    _step[0] = ""
    log(f"\n{'=' * 78}\nSTEP {n} — {title}\n{'=' * 78}")
    _step[0] = "  "


def check(label, ok, measured):
    """Print what was measured, then whether it passed. A bare OK proves nothing."""
    log(f"[{'OK  ' if ok else 'FAIL'}] {label}: {measured}")
    if not ok:
        raise SystemExit(f"\nSTOPPED: {label} — measured {measured}")


# ----------------------------------------------------------------------------------------
def step1_population():
    """Read the four per-isoform arrays, select test_clean, and fix the isoform order."""
    step(1, "define the population")
    with h5py.File(H5, "r") as f:
        dec   = lambda a: np.array([x.decode() if isinstance(x, bytes) else x for x in a])
        split = dec(f["split"][:])
        chrom = dec(f["chr"][:])
        y_all = f["labels"][:]
        ids_a = dec(f["isoform_id"][:])
    log(f"file: {H5}")
    check("arrays share one length", len({len(split), len(chrom), len(y_all), len(ids_a)}) == 1,
          f"{len(split):,} entries")

    keep = np.where(split == SPLIT)[0]
    n, y, ids = len(keep), y_all[keep].astype(int), ids_a[keep]

    check("every isoform is on a held-out chromosome",
          set(chrom[keep]) <= {"chr1", "chr3", "chr5", "chr7"},
          sorted(set(chrom[keep])))
    check("no test_paralog isoform is included",
          not (split[keep] == "test_paralog").any(),
          f"{(split[keep] == 'test_paralog').sum()} present")
    check("n matches the plan", n == 10520, f"{n:,}")
    check("indices are strictly increasing (required for HDF5 reads)",
          bool((np.diff(keep) > 0).all()), "yes" if (np.diff(keep) > 0).all() else "no")

    log(f"      NMD susceptible: {y.sum():,} of {n:,}  ({100 * y.mean():.1f}%)")
    log(f"      alignment key  : isoform_id, first = {ids[0]}, last = {ids[-1]}")
    return dict(keep=keep, n=n, y=y, ids=ids)


# ----------------------------------------------------------------------------------------
def step2_resamples(pop):
    """Draw the bootstrap resamples ONCE, shared by both configurations and both metrics."""
    step(2, "draw the bootstrap resamples")
    n, y = pop["n"], pop["y"]
    rng = np.random.default_rng(RSEED)
    R = rng.integers(0, n, size=(B, n))
    pos = y[R].sum(axis=1)
    check("resample count", R.shape == (B, n), f"{B:,} resamples of {n:,} isoforms")
    check("both classes present in every resample",
          bool(((pos > 0) & (pos < n)).all()),
          f"NMD per resample ranges {pos.min():,}–{pos.max():,} (observed {y.sum():,})")
    log(f"      seed {RSEED}, shared across both configurations and both metrics")
    return R


# ----------------------------------------------------------------------------------------
def _widths(tag):
    m = re.fullmatch(r"atg(\d+)_stop(\d+)", tag)
    return int(m.group(1)), int(m.group(2))


def step3_score(tag, pop, cfg):
    """Score the five members. Widths come from the TAG, never from the checkpoint's config."""
    step(3, f"score the five members — {tag}")
    ws_atg, ws_stop = _widths(tag)
    keep, ids = pop["keep"], pop["ids"]
    log(f"widths from tag: start {ws_atg}, stop {ws_stop}")

    L, meta = [], []
    with h5py.File(H5, "r") as f:
        mean = f["normalization/orf_feat_mean"][:].astype(np.float32)
        std  = f["normalization/orf_feat_std"][:].astype(np.float32)
        log(f"feature normalization applied: mean {np.round(mean, 3)}  sd {np.round(std, 3)}")

        for s in SEEDS:
            ck_path = CKPT_DIR / f"best_model_{tag}_seed{s}.pt"
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            model = build_model({**cfg["model"],
                                 "window_size_atg": ws_atg, "window_size_stop": ws_stop})
            model.load_state_dict(ck["model_state_dict"], strict=True)
            model.eval()                      # batch-norm uses stored running statistics
            check(f"seed {s} loaded, eval mode, buffers present",
                  not model.training and any("running_mean" in k for k in ck["model_state_dict"]),
                  f"{len(ck['model_state_dict'])} tensors, epoch {ck['epoch']}, "
                  f"stored val_auc {ck['val_auc']:.5f}")

            out = []
            for i in range(0, len(keep), BATCH):
                idx  = keep[i:i + BATCH]
                atg  = f[f"w{ws_atg}/atg_windows"][idx].astype(np.float32)
                stp  = f[f"w{ws_stop}/stop_windows"][idx].astype(np.float32)
                feat = (f["orf_features"][idx].astype(np.float32) - mean) / std
                mask = f["orf_mask"][idx]
                with torch.no_grad():
                    lg = model(torch.from_numpy(atg), torch.from_numpy(stp),
                               torch.from_numpy(feat), torch.from_numpy(mask))
                out.append(lg.squeeze(-1).numpy())
            v = np.concatenate(out)
            check(f"seed {s} scored every isoform, in order", len(v) == pop["n"],
                  f"{len(v):,} log-odds, range {v.min():.3f} to {v.max():.3f}")
            L.append(v)
            meta.append(dict(seed=s, epoch=int(ck["epoch"]), stored_val_auc=float(ck["val_auc"])))

    L = np.vstack(L)
    if len(SEEDS) < 2:
        raise SystemExit("the member-distinctness check needs at least 2 seeds")
    pair_max = max(np.abs(L[i] - L[j]).max()
                   for i in range(len(SEEDS)) for j in range(i + 1, len(SEEDS)))
    check("the five members are five different models",
          pair_max > 0,
          f"largest per-isoform disagreement between any two members = {pair_max:.4f} log-odds")
    return L, meta


# ----------------------------------------------------------------------------------------
def step4_estimates(tag, pop, L):
    """Ensemble = mean log-odds. Member scores give the spread across fitting runs."""
    step(4, f"ensemble and point estimates — {tag}")
    y = pop["y"]
    ens = L.mean(axis=0)
    check("ensemble formed on the log-odds scale", True,
          f"mean log-odds range {ens.min():.3f} to {ens.max():.3f}")

    auc_e, ap_e = roc_auc_score(y, ens), average_precision_score(y, ens)
    log(f"      ENSEMBLE  AUC {auc_e:.5f}   AUPRC {ap_e:.5f}")

    auc_m = np.array([roc_auc_score(y, L[i]) for i in range(len(SEEDS))])
    ap_m  = np.array([average_precision_score(y, L[i]) for i in range(len(SEEDS))])
    for s, a, p in zip(SEEDS, auc_m, ap_m):
        log(f"        seed {s}:  AUC {a:.5f}   AUPRC {p:.5f}")
    log(f"      member mean  AUC {auc_m.mean():.5f} (sd {auc_m.std(ddof=1):.5f})   "
        f"AUPRC {ap_m.mean():.5f} (sd {ap_m.std(ddof=1):.5f})")
    return dict(ens=ens, auc_ens=float(auc_e), auprc_ens=float(ap_e),
                auc_members=auc_m.tolist(), auprc_members=ap_m.tolist(),
                auc_member_mean=float(auc_m.mean()), auc_member_sd=float(auc_m.std(ddof=1)),
                auprc_member_mean=float(ap_m.mean()), auprc_member_sd=float(ap_m.std(ddof=1)))


# ----------------------------------------------------------------------------------------
def step5_interval(tag, pop, res, R):
    """Resample the test set with the model held fixed."""
    step(5, f"bootstrap interval on the ensemble — {tag}")
    y, ens = pop["y"], res["ens"]
    ba = np.empty(B); bp = np.empty(B)
    for b in range(B):
        idx = R[b]
        ba[b] = roc_auc_score(y[idx], ens[idx])
        bp[b] = average_precision_score(y[idx], ens[idx])
        if (b + 1) % 500 == 0:
            log(f"      {b + 1:,}/{B:,} resamples")
    ci_a = np.percentile(ba, [2.5, 97.5]); ci_p = np.percentile(bp, [2.5, 97.5])
    log(f"      AUC   {res['auc_ens']:.5f}  95% CI [{ci_a[0]:.5f}, {ci_a[1]:.5f}]")
    log(f"      AUPRC {res['auprc_ens']:.5f}  95% CI [{ci_p[0]:.5f}, {ci_p[1]:.5f}]")
    res.update(auc_ci=[float(ci_a[0]), float(ci_a[1])],
               auprc_ci=[float(ci_p[0]), float(ci_p[1])])
    return res


# ----------------------------------------------------------------------------------------
def step6_sensitivity(tag, pop, L, res):
    """Combine members on the probability scale instead. Declared sensitivity check."""
    step(6, f"aggregation sensitivity — {tag}")
    y = pop["y"]
    ens_p = 1.0 / (1.0 + np.exp(-L))          # sigmoid per member
    ens_p = ens_p.mean(axis=0)                 # mean PROBABILITY, not mean log-odds
    a, p = roc_auc_score(y, ens_p), average_precision_score(y, ens_p)
    log(f"      mean-probability  AUC {a:.5f}   AUPRC {p:.5f}")
    log(f"      difference from primary (mean log-odds): "
        f"AUC {a - res['auc_ens']:+.5f}   AUPRC {p - res['auprc_ens']:+.5f}")
    res.update(auc_alt=float(a), auprc_alt=float(p))
    return res


# ----------------------------------------------------------------------------------------
def step7_write(tag, pop, L, res, meta):
    step(7, f"write — {tag}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{tag}_ens5_test_clean"

    metrics = dict(
        tag=tag, split="test_clean", evaluation_class="final_test",
        aggregation="mean_logit", n=int(pop["n"]), n_nmd=int(pop["y"].sum()),
        members=meta, bootstrap=dict(seed=RSEED, resamples=B, unit="isoform"),
        written_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **{k: v for k, v in res.items() if k != "ens"})
    mp = OUT_DIR / f"ensemble_metrics_{stem}.json"
    mp.write_text(json.dumps(metrics, indent=2))
    log(f"wrote {mp}  ({mp.stat().st_size:,} bytes)")

    cols = ["isoform_id", "label"] + [f"logit_seed{s}" for s in SEEDS] + ["logit_ensemble"]
    pp = OUT_DIR / f"ensemble_predictions_{stem}.tsv"
    with open(pp, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for i in range(pop["n"]):
            fh.write("\t".join([pop["ids"][i], str(pop["y"][i])]
                               + [f"{L[k][i]:.6f}" for k in range(len(SEEDS))]
                               + [f"{res['ens'][i]:.6f}"]) + "\n")
    log(f"wrote {pp}  ({pp.stat().st_size:,} bytes, {pop['n']:,} rows)")


# ----------------------------------------------------------------------------------------
def main():
    log(f"Analysis 1 — held-out performance.  started {datetime.now().isoformat(timespec='seconds')}")
    log(f"torch {torch.__version__}, numpy {np.__version__}")
    cfg = yaml.safe_load(CONFIG.read_text())

    pop = step1_population()
    R   = step2_resamples(pop)

    for tag in TAGS:
        L, meta = step3_score(tag, pop, cfg)
        res = step4_estimates(tag, pop, L)
        res = step5_interval(tag, pop, res, R)
        res = step6_sensitivity(tag, pop, L, res)
        step7_write(tag, pop, L, res, meta)

    _step[0] = ""
    log(f"\nDone. {datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()

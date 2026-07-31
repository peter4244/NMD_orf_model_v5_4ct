#!/usr/bin/env python3
"""
probe_deepshap_additivity.py — does DeepSHAP's completeness hold on THIS architecture?

WHY. ANALYSIS4_PLAN proposes attributing per-nucleotide importance with DeepSHAP, because a
1,000-2,000 position window cannot be enumerated the way Analyses 2 and 3 were. Two independent
reviews raised the same concern about whether the method is even valid here:

  shap's PyTorch DeepExplainer attaches its DeepLIFT rules by walking nn.Module instances. In
  this architecture the dominant nonlinearities are NOT modules -- model.py uses functional
  F.relu after each conv/batchnorm, a global .max(dim=-1) pool, and a masked softmax with bmm
  inside the attention aggregator. If the rules do not attach, DeepSHAP degenerates toward
  gradient x (input - baseline) and its completeness error may be large and systematic.

  Circumstantial support that someone already hit this: deepshap.py passes
  check_additivity=False at ALL THREE call sites (:352, :438, :538).

WHAT THIS MEASURES, and it deliberately does not trust shap's own verdict alone:

  1. Whether shap.DeepExplainer(...).shap_values(..., check_additivity=True) RAISES.
  2. Any warning shap emits about unhandled module types -- captured, not swallowed.
  3. THE COMPLETENESS ERROR COMPUTED INDEPENDENTLY, per isoform:
         sum over positions and channels of phi   vs   f(x) - mean over background of f(bg)
     Both terms are evaluated here by running the wrapper directly. This is the number that
     decides the design, and it does not depend on shap agreeing that there is a problem.

WHAT THE ANSWER DECIDES. If the error is small relative to the branch effect, ANALYSIS4 proceeds
with DeepSHAP and the corrections the reviews identified. If it is large, the positional analysis
is built on occlusion -- replace high-attribution positions with reference values and measure the
actual change in the logit -- which uses the model directly and is immune to this entire class of
concern.

Run on Explorer (shap is not installed in the local env):
    python analysis_plans/probe_deepshap_additivity.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shap                                                   # noqa: E402
from deepshap import BranchWrapper                            # noqa: E402
from model import build_model                                 # noqa: E402
from utils import NMDDataset, load_config, resolve_checkpoint, split_indices  # noqa: E402

CONFIG   = "config_dn.yaml"
CKPT_DIR = "results_4ct_sweep"
TAG      = "atg1000_stop1000"
MEMBER   = 100
DRAW     = 1
N_BG     = 100          # small: this probe is about correctness, not about the final estimate
N_EXPL   = 20
BRANCHES = ["atg", "stop"]


def main():
    sys.stdout.reconfigure(line_buffering=True)
    print(f"shap {shap.__version__}   torch {torch.__version__}   numpy {np.__version__}")
    cfg = load_config(CONFIG)
    h5 = cfg["data"]["hdf5_path"]
    wa = int(TAG.split("_")[0].replace("atg", ""))
    ws = int(TAG.split("_")[1].replace("stop", ""))
    device = torch.device("cpu")

    model = build_model({**cfg["model"], "window_size_atg": wa, "window_size_stop": ws}).to(device)
    ck = resolve_checkpoint(CKPT_DIR, TAG, MEMBER)
    model.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["model_state_dict"])
    model.eval()
    print(f"model {ck}")

    print("\n=== which nonlinearities are nn.Modules, and which are functional? ===")
    mods = [type(m).__name__ for m in model.modules()]
    from collections import Counter
    print(f"  module types present: {dict(Counter(mods))}")
    print("  NOTE: F.relu, .max(dim=-1), the masked softmax and bmm are NOT in that list by")
    print("        construction -- they are functional calls in model.py, not modules.")

    n_train = len(split_indices(h5, "train"))
    bg_pos = np.random.RandomState(42 + 1000 * DRAW).choice(n_train, N_BG, replace=False)
    train_ds = NMDDataset(h5, wa, ws, split="train", restrict_to=bg_pos)
    expl_ds = NMDDataset(h5, wa, ws, split="all", restrict_to=np.arange(N_EXPL))

    for branch in BRANCHES:
        print(f"\n{'=' * 74}\nBRANCH: {branch}\n{'=' * 74}")
        key = f"{branch}_windows"
        bg = torch.stack([train_ds[i][key][0] for i in range(len(train_ds))]).to(device)
        print(f"  background {tuple(bg.shape)}")

        raised = None
        errs, deltas = [], []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for i in range(len(expl_ds)):
                s = expl_ds[i]
                wrap = BranchWrapper(
                    model, branch=branch,
                    fixed_atg_windows=s["atg_windows"].unsqueeze(0).to(device),
                    fixed_stop_windows=s["stop_windows"].unsqueeze(0).to(device),
                    fixed_orf_features=s["orf_features"].unsqueeze(0).to(device),
                    fixed_orf_mask=s["orf_mask"].unsqueeze(0).to(device),
                    orf_index=0)
                x = s[key][0].unsqueeze(0).to(device)
                ex = shap.DeepExplainer(wrap, bg)

                if i == 0:
                    # Does shap ITSELF think completeness holds? Ask once, loudly.
                    try:
                        ex.shap_values(x, check_additivity=True)
                        raised = False
                    except Exception as e:
                        raised = True
                        print(f"  check_additivity=True RAISED: {type(e).__name__}")
                        print(f"    {str(e)[:400]}")

                sv = ex.shap_values(x, check_additivity=False)
                if isinstance(sv, list):
                    sv = sv[0]
                if torch.is_tensor(sv):
                    sv = sv.cpu().numpy()
                sv = np.asarray(sv)
                if sv.ndim == 4 and sv.shape[-1] == 1:
                    sv = sv[..., 0]

                # THE INDEPENDENT MEASUREMENT. Both terms evaluated by running the wrapper.
                with torch.no_grad():
                    fx = float(wrap(x).item())
                    ebg = float(wrap(bg).mean().item())
                errs.append(float(sv.sum()) - (fx - ebg))
                deltas.append(fx - ebg)

        if raised is False:
            print("  check_additivity=True did NOT raise")
        mods_warned = [str(w.message)[:160] for w in caught
                       if "unrecognized" in str(w.message).lower()
                       or "nn.Module" in str(w.message)
                       or "not supported" in str(w.message).lower()]
        print(f"  shap warnings about unhandled ops: {len(mods_warned)}")
        for m in dict.fromkeys(mods_warned):
            print(f"    {m}")

        e = np.abs(np.array(errs)); d = np.abs(np.array(deltas))
        print(f"\n  COMPLETENESS ERROR, measured independently over {len(errs)} isoforms:")
        print(f"    |sum(phi) - (f(x) - E_bg[f])|   median {np.median(e):.4f}   max {e.max():.4f}")
        print(f"    |f(x) - E_bg[f]|  (the effect)  median {np.median(d):.4f}   max {d.max():.4f}")
        rel = np.median(e) / np.median(d) if np.median(d) > 0 else float("inf")
        print(f"    median error as a fraction of the median effect: {100 * rel:.1f}%")
        print(f"    VERDICT: {'DeepSHAP is usable here' if rel < 0.20 else 'ERROR IS LARGE -- positional analysis should not rest on DeepSHAP'}")
        print(f"    (pre-registered threshold: 20% of the median effect)")


if __name__ == "__main__":
    main()

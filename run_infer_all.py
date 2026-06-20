#!/usr/bin/env python3
"""
run_infer_all.py — Score every isoform in the model's H5 dataset (split="all")
                   using the trained NMD ORF model checkpoint.

Drop-in replacement for evaluate.py's prediction loop, but with
NMDDataset(split="all") so train+val+test isoforms all get predictions
written to a single TSV. Used to broaden the head-to-head model
comparison from the 561-isoform test-set intersection to the full
intersection between our n=1,166 ref-AUG-traceable cohort and the
model's H5 universe.

Usage (on Explorer, from ~/cc/nmd_orf_model_v5_4ct):
    conda activate nmd_model
    python run_infer_all.py --config config.yaml \
        --atg-window 500 --stop-window 500

Output:
    results_4ct/predictions_all_atg500_stop500.tsv   # one row per isoform
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

# Allow running from anywhere — assume model repo on sys.path
HERE = Path(__file__).resolve().parent
REPO = HERE
for cand in (HERE, HERE.parent, Path("~/cc/nmd_orf_model_v5_4ct").expanduser()):
    if (cand / "model.py").exists() and (cand / "utils.py").exists():
        REPO = cand
        break
sys.path.insert(0, str(REPO))

from model import build_model
from utils import NMDDataset, load_config, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO / "config.yaml"))
    p.add_argument("--atg-window", type=int, default=None)
    p.add_argument("--stop-window", type=int, default=None)
    p.add_argument("--results-dir", default=None,
                   help="override; default reads config['data']['results_dir'] or results_4ct")
    p.add_argument("--split", default="all",
                   help="NMDDataset split (default: 'all')")
    args = p.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ws_atg = args.atg_window or config["data"]["window_size_atg"]
    ws_stop = args.stop_window or config["data"]["window_size_stop"]
    tag = f"atg{ws_atg}_stop{ws_stop}"
    results_dir = Path(args.results_dir or "results_4ct")
    if not results_dir.is_absolute():
        results_dir = REPO / results_dir

    ckpt_path = results_dir / f"best_model_{tag}.pt"
    print(f"[infer-all] checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_config = {**config["model"],
                    "window_size_atg": ws_atg, "window_size_stop": ws_stop}
    model = build_model(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[infer-all] trained {ckpt['epoch']} epochs, val AUC = {ckpt['val_auc']:.4f}")

    h5_path = config["data"]["hdf5_path"]
    ds = NMDDataset(h5_path, ws_atg, ws_stop, split=args.split)
    loader = DataLoader(ds, batch_size=config["training"]["batch_size"],
                        shuffle=False, num_workers=0)

    use_amp = bool(config["training"].get("use_amp", False))

    all_labels, all_logits = [], []
    with torch.no_grad():
        for batch in loader:
            atg = batch["atg_windows"].to(device)
            stop = batch["stop_windows"].to(device)
            orf_feat = batch["orf_features"].to(device)
            mask = batch["orf_mask"].to(device)
            if use_amp and device.type == "cuda":
                with autocast(device_type="cuda"):
                    cls_logits, _ = model(atg, stop, orf_feat, mask,
                                           return_attention=True)
            else:
                cls_logits, _ = model(atg, stop, orf_feat, mask,
                                       return_attention=True)
            all_labels.extend(batch["label"].numpy())
            all_logits.extend(cls_logits.squeeze(-1).cpu().numpy())

    labels = np.array(all_labels)
    logits = np.array(all_logits)
    probs = torch.sigmoid(torch.tensor(logits)).numpy()

    with h5py.File(h5_path, "r") as f:
        ids = np.array([s.decode() if isinstance(s, bytes) else s
                        for s in f["isoform_id"][:]])
        chrs = np.array([s.decode() if isinstance(s, bytes) else s
                         for s in f["chr"][:]])
        splits = np.array([s.decode() if isinstance(s, bytes) else s
                           for s in f["split"][:]])
    sel_ids = ids[ds.indices]
    sel_chrs = chrs[ds.indices]
    sel_splits = splits[ds.indices]

    df = pd.DataFrame({
        "isoform_id": sel_ids,
        "chr":         sel_chrs,
        "h5_split":    sel_splits,
        "label":       labels.astype(int),
        "logit":       logits,
        "prob":        probs,
    })
    out_path = results_dir / f"predictions_{args.split}_{tag}.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"[infer-all] wrote {out_path}  (n={len(df):,})")

    n_by_split = df["h5_split"].value_counts().to_dict()
    print(f"[infer-all] rows per H5 split: {n_by_split}")
    print(f"[infer-all] label=1 (NMD+): {int((df['label']==1).sum()):,}; "
          f"label=0 (Control): {int((df['label']==0).sum()):,}")


if __name__ == "__main__":
    main()

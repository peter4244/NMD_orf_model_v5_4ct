"""
utils.py — Shared utilities for NMD ORF model v5: dataset, metrics, config loading.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import Dataset


def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NMDDataset(Dataset):
    """
    v5: PyTorch dataset — loads windows + minimal ORF features from HDF5.
    No tx_features (removed in v5).
    """

    def __init__(self, h5_path, window_size_atg, window_size_stop, split="train"):
        self.h5_path = Path(h5_path)

        with h5py.File(self.h5_path, "r") as f:
            splits = np.array([s.decode() if isinstance(s, bytes) else s
                               for s in f["split"][:]])
            if split == "test_clean":
                mask = splits == "test"
            elif split == "test_all":
                mask = (splits == "test") | (splits == "test_paralog")
            else:
                mask = splits == split

            self.indices = np.where(mask)[0]
            n = len(self.indices)

            # Load window data into RAM
            self.atg_windows = f[f"w{window_size_atg}"]["atg_windows"][self.indices].astype(np.float32)
            self.stop_windows = f[f"w{window_size_stop}"]["stop_windows"][self.indices].astype(np.float32)

            self.orf_features = f["orf_features"][self.indices].astype(np.float32)
            self.orf_mask = f["orf_mask"][self.indices]
            self.labels = f["labels"][self.indices].astype(np.float32)

            # Load normalization stats
            self.orf_feat_mean = f["normalization/orf_feat_mean"][:].astype(np.float32)
            self.orf_feat_std = f["normalization/orf_feat_std"][:].astype(np.float32)

        # Apply normalization to ORF features
        self.orf_features = (self.orf_features - self.orf_feat_mean) / self.orf_feat_std

        # Zero out normalized features for padded ORFs
        self.orf_features[~self.orf_mask] = 0.0

        print(f"  {split}: {n:,} samples, {self.labels.sum():.0f} NMD+ "
              f"({self.labels.mean()*100:.1f}%)")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return {
            "atg_windows": torch.from_numpy(self.atg_windows[idx]),
            "stop_windows": torch.from_numpy(self.stop_windows[idx]),
            "orf_features": torch.from_numpy(self.orf_features[idx]),
            "orf_mask": torch.from_numpy(self.orf_mask[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


def compute_pos_weight(dataset):
    """Compute pos_weight for BCEWithLogitsLoss from label distribution."""
    n_pos = dataset.labels.sum()
    n_neg = len(dataset.labels) - n_pos
    return torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32)


def compute_metrics(labels, logits):
    """Compute AUC and AUPRC from labels and raw logits."""
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    labels = np.array(labels)
    auc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    return {"auc": auc, "auprc": auprc}

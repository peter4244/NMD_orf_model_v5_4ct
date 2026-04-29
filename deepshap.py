#!/usr/bin/env python3
"""
deepshap.py — DeepSHAP sequence interpretation for NMD ORF model.

Creates wrapper models isolating ATG and stop CNN branches,
runs DeepExplainer to get per-position SHAP values, saves results.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn

from model import NMDOrfModel, build_model
from utils import NMDDataset, load_config, set_seed


class BranchWrapper(nn.Module):
    """
    Wraps a single sequence branch (ATG or stop) of the ORF model.
    Fixes all other inputs at observed values, varies only the target window.

    Input: (batch, n_channels, window_size)
    Output: (batch, 1) — classification logit
    """

    def __init__(self, full_model, branch="atg",
                 fixed_atg_windows=None, fixed_stop_windows=None,
                 fixed_orf_features=None, fixed_orf_mask=None,
                 orf_index=0):
        super().__init__()
        self.full_model = full_model
        self.branch = branch
        self.fixed_atg_windows = fixed_atg_windows    # (1, K, C, W)
        self.fixed_stop_windows = fixed_stop_windows  # (1, K, C, W)
        self.fixed_orf_features = fixed_orf_features
        self.fixed_orf_mask = fixed_orf_mask
        self.orf_index = orf_index

    def forward(self, x):
        """x: (batch, n_channels, window_size) — the window being explained."""
        batch_size = x.shape[0]
        device = x.device

        # Expand fixed inputs to match batch size
        atg_win = self.fixed_atg_windows.expand(batch_size, -1, -1, -1).clone().to(device)
        stop_win = self.fixed_stop_windows.expand(batch_size, -1, -1, -1).clone().to(device)
        orf_feat = self.fixed_orf_features.expand(batch_size, -1, -1).to(device)
        mask = self.fixed_orf_mask.expand(batch_size, -1).to(device)

        # Replace only the target ORF's window in the correct branch
        if self.branch == "atg":
            atg_win[:, self.orf_index] = x
        else:
            stop_win[:, self.orf_index] = x

        cls_logits = self.full_model(atg_win, stop_win, orf_feat, mask)
        return cls_logits


class StructuralBranchWrapper(nn.Module):
    """
    Wraps the structural feature branch for the rank-0 ORF.
    Fixes all sequence windows at observed values, varies only orf_features.

    Input: (batch, n_orf_features) — the 5 structural features
    Output: (batch, 1) — classification logit
    """

    def __init__(self, full_model,
                 fixed_atg_windows=None, fixed_stop_windows=None,
                 fixed_orf_features=None, fixed_orf_mask=None,
                 orf_index=0):
        super().__init__()
        self.full_model = full_model
        self.fixed_atg_windows = fixed_atg_windows    # (1, K, C, W)
        self.fixed_stop_windows = fixed_stop_windows  # (1, K, C, W)
        self.fixed_orf_features = fixed_orf_features   # (1, K, F)
        self.fixed_orf_mask = fixed_orf_mask
        self.orf_index = orf_index

    def forward(self, x):
        """x: (batch, n_orf_features) — structural features for the target ORF."""
        batch_size = x.shape[0]
        device = x.device

        atg_win = self.fixed_atg_windows.expand(batch_size, -1, -1, -1).to(device)
        stop_win = self.fixed_stop_windows.expand(batch_size, -1, -1, -1).to(device)
        orf_feat = self.fixed_orf_features.expand(batch_size, -1, -1).clone().to(device)
        mask = self.fixed_orf_mask.expand(batch_size, -1).to(device)

        # Replace only the target ORF's structural features
        orf_feat[:, self.orf_index] = x

        cls_logits = self.full_model(atg_win, stop_win, orf_feat, mask)
        return cls_logits


class JointBranchWrapper(nn.Module):
    """
    Wraps ALL rank-0 ORF inputs (ATG window + stop window + structural features)
    as a single flattened input vector. Fixes non-rank-0 ORF context at observed values.

    This gives additive SHAP values: sum(SHAP) = f(x) - E[f(x)] across all inputs.

    Input: (batch, C*W_atg + C*W_stop + F) — concatenated rank-0 inputs
    Output: (batch, 1) — classification logit

    Layout of the flattened input:
      [0 : C*W_atg]                    = ATG window (C channels × W_atg positions)
      [C*W_atg : C*W_atg + C*W_stop]   = Stop window (C channels × W_stop positions)
      [C*W_atg + C*W_stop : end]        = Structural features (F values)
    """

    def __init__(self, full_model, n_channels, w_atg, w_stop, n_features,
                 fixed_atg_windows=None, fixed_stop_windows=None,
                 fixed_orf_features=None, fixed_orf_mask=None,
                 orf_index=0):
        super().__init__()
        self.full_model = full_model
        self.n_channels = n_channels
        self.w_atg = w_atg
        self.w_stop = w_stop
        self.n_features = n_features
        self.atg_size = n_channels * w_atg
        self.stop_size = n_channels * w_stop
        self.fixed_atg_windows = fixed_atg_windows    # (1, K, C, W_atg)
        self.fixed_stop_windows = fixed_stop_windows  # (1, K, C, W_stop)
        self.fixed_orf_features = fixed_orf_features   # (1, K, F)
        self.fixed_orf_mask = fixed_orf_mask           # (1, K)
        self.orf_index = orf_index

    def forward(self, x):
        """x: (batch, C*W_atg + C*W_stop + F) — flattened rank-0 inputs."""
        batch_size = x.shape[0]
        device = x.device

        # Unpack the flattened input
        atg_flat = x[:, :self.atg_size]
        stop_flat = x[:, self.atg_size:self.atg_size + self.stop_size]
        struct_flat = x[:, self.atg_size + self.stop_size:]

        # Reshape sequence windows: (batch, C, W)
        atg_rank0 = atg_flat.reshape(batch_size, self.n_channels, self.w_atg)
        stop_rank0 = stop_flat.reshape(batch_size, self.n_channels, self.w_stop)

        # Expand fixed context for non-rank-0 ORFs
        atg_win = self.fixed_atg_windows.expand(batch_size, -1, -1, -1).clone().to(device)
        stop_win = self.fixed_stop_windows.expand(batch_size, -1, -1, -1).clone().to(device)
        orf_feat = self.fixed_orf_features.expand(batch_size, -1, -1).clone().to(device)
        mask = self.fixed_orf_mask.expand(batch_size, -1).to(device)

        # Replace rank-0 ORF's inputs with the varied values
        atg_win[:, self.orf_index] = atg_rank0
        stop_win[:, self.orf_index] = stop_rank0
        orf_feat[:, self.orf_index] = struct_flat

        cls_logits = self.full_model(atg_win, stop_win, orf_feat, mask)
        return cls_logits


def run_deepshap(config_path="config.yaml", n_explain=2000, n_background=100,
                  atg_window=None, stop_window=None, seed=None, run_id=None,
                  branches=None, orf_index=0):
    config = load_config(config_path)
    seed = seed if seed is not None else config["training"]["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ws_atg = atg_window or config["data"]["window_size_atg"]
    ws_stop = stop_window or config["data"]["window_size_stop"]
    tag = f"atg{ws_atg}_stop{ws_stop}"
    run_suffix = f"_run{run_id}" if run_id is not None else ""
    orf_suffix = f"_orf{orf_index}" if orf_index != 0 else ""
    results_dir = Path("results")
    h5_path = config["data"]["hdf5_path"]

    # Load model
    ckpt_path = results_dir / f"best_model_{tag}.pt"
    print(f"Loading model from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_config = {**config["model"],
                    "window_size_atg": ws_atg, "window_size_stop": ws_stop}
    model = build_model(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load data
    print("Loading test and training data ...")
    test_ds = NMDDataset(h5_path, ws_atg, ws_stop, split="test_clean")
    train_ds = NMDDataset(h5_path, ws_atg, ws_stop, split="train")

    print(f"ORF index: {orf_index}")

    # Subsample for tractability (n_explain=0 means all test samples)
    rng = np.random.RandomState(seed)
    if n_explain == 0 or n_explain >= len(test_ds):
        explain_idx = np.arange(len(test_ds))
    else:
        explain_idx = rng.choice(len(test_ds), size=n_explain, replace=False)
    bg_idx = rng.choice(len(train_ds), size=min(n_background, len(train_ds)),
                        replace=False)

    # Which branches to run
    if branches is None:
        branches = ["atg", "stop", "structural"]
    valid_branches = {"atg", "stop", "structural", "joint"}
    print(f"Explaining {len(explain_idx)} test samples, {len(bg_idx)} background")
    print(f"Branches: {branches}")

    seq_branches = [b for b in branches if b in ("atg", "stop")]
    run_structural = "structural" in branches
    run_joint = "joint" in branches
    batch_size = 100

    for branch in seq_branches:
        print(f"\n{'='*40}")
        print(f"DeepSHAP for {branch.upper()} windows")
        print(f"{'='*40}")

        # Build background tensor (target ORF's window only)
        # Filter to samples where this ORF rank is present
        bg_idx_masked = [i for i in bg_idx if train_ds[i]["orf_mask"][orf_index]]
        if branch == "atg":
            bg_data = np.stack([train_ds[i]["atg_windows"][orf_index].numpy()
                                for i in bg_idx_masked])
        else:
            bg_data = np.stack([train_ds[i]["stop_windows"][orf_index].numpy()
                                for i in bg_idx_masked])
        bg_tensor = torch.tensor(bg_data, dtype=torch.float32).to(device)

        # Filter explain samples to those with this ORF rank present
        seq_explain_idx = np.array([i for i in explain_idx
                                    if test_ds[i]["orf_mask"][orf_index]])
        if len(seq_explain_idx) < len(explain_idx):
            print(f"  Filtered to {len(seq_explain_idx)}/{len(explain_idx)} "
                  f"samples with ORF rank {orf_index} present")

        # Explain in batches — each sample uses its own context
        all_shap = []
        all_inputs = []
        all_labels = []

        for start in range(0, len(seq_explain_idx), batch_size):
            end = min(start + batch_size, len(seq_explain_idx))
            batch_idx = seq_explain_idx[start:end]

            batch_shap = []
            for i in batch_idx:
                sample = test_ds[i]

                # Create wrapper with this sample's own context
                wrapper = BranchWrapper(
                    model, branch=branch,
                    fixed_atg_windows=sample["atg_windows"].unsqueeze(0).to(device),
                    fixed_stop_windows=sample["stop_windows"].unsqueeze(0).to(device),
                    fixed_orf_features=sample["orf_features"].unsqueeze(0).to(device),
                    fixed_orf_mask=sample["orf_mask"].unsqueeze(0).to(device),
                    orf_index=orf_index,
                )

                explainer = shap.DeepExplainer(wrapper, bg_tensor)

                if branch == "atg":
                    input_win = sample["atg_windows"][orf_index].unsqueeze(0).to(device)
                else:
                    input_win = sample["stop_windows"][orf_index].unsqueeze(0).to(device)

                sv = explainer.shap_values(input_win, check_additivity=False)
                if isinstance(sv, list):
                    sv = sv[0]
                if torch.is_tensor(sv):
                    sv = sv.cpu().numpy()
                batch_shap.append(sv[0])  # (C, W)

            if branch == "atg":
                inputs = np.stack([test_ds[i]["atg_windows"][orf_index].numpy()
                                   for i in batch_idx])
            else:
                inputs = np.stack([test_ds[i]["stop_windows"][orf_index].numpy()
                                   for i in batch_idx])
            labels = np.array([test_ds[i]["label"].item() for i in batch_idx])

            all_shap.append(np.stack(batch_shap))
            all_inputs.append(inputs)
            all_labels.append(labels)

            print(f"  Explained {end}/{len(seq_explain_idx)} samples")

        shap_arr = np.concatenate(all_shap, axis=0)
        input_arr = np.concatenate(all_inputs, axis=0)
        label_arr = np.concatenate(all_labels, axis=0)

        # Save
        out_path = results_dir / f"deepshap_{branch}_{tag}{orf_suffix}{run_suffix}.npz"
        np.savez_compressed(out_path,
                            shap_values=shap_arr,
                            inputs=input_arr,
                            labels=label_arr,
                            explain_indices=seq_explain_idx,
                            channel_names=["A", "C", "G", "T", "junction",
                                          "rolling_gc", "frame_0", "frame_1",
                                          "frame_2"],
                            seed=np.array(seed))
        print(f"  -> Saved {out_path} ({shap_arr.shape})")

    # --- Structural branch ---
    with h5py.File(h5_path, 'r') as f:
        orf_feat_names = json.loads(f.attrs["orf_feature_cols"])

    if not run_structural:
        print("\nSkipping structural branch (not in requested branches)")
    else:
        print(f"\n{'='*40}")
        print(f"DeepSHAP for STRUCTURAL features ({len(orf_feat_names)} features)")
        print(f"{'='*40}")

        # Background: ORF structural features from training samples
        bg_idx_masked_s = [i for i in bg_idx if train_ds[i]["orf_mask"][orf_index]]
        bg_struct = np.stack([train_ds[i]["orf_features"][orf_index].numpy()
                              for i in bg_idx_masked_s])
        bg_struct_tensor = torch.tensor(bg_struct, dtype=torch.float32).to(device)

        # Filter to samples where this ORF rank is present
        struct_explain_idx = np.array([i for i in explain_idx
                                       if test_ds[i]["orf_mask"][orf_index]])
        if len(struct_explain_idx) < len(explain_idx):
            print(f"  Filtered to {len(struct_explain_idx)}/{len(explain_idx)} "
                  f"samples with ORF rank {orf_index} present")

        all_shap_s = []
        all_inputs_s = []
        all_labels_s = []

        for start in range(0, len(struct_explain_idx), batch_size):
            end = min(start + batch_size, len(struct_explain_idx))
            batch_idx = struct_explain_idx[start:end]

            batch_shap = []
            for i in batch_idx:
                sample = test_ds[i]

                wrapper = StructuralBranchWrapper(
                    model,
                    fixed_atg_windows=sample["atg_windows"].unsqueeze(0).to(device),
                    fixed_stop_windows=sample["stop_windows"].unsqueeze(0).to(device),
                    fixed_orf_features=sample["orf_features"].unsqueeze(0).to(device),
                    fixed_orf_mask=sample["orf_mask"].unsqueeze(0).to(device),
                    orf_index=orf_index,
                )

                explainer = shap.DeepExplainer(wrapper, bg_struct_tensor)
                input_feat = sample["orf_features"][orf_index].unsqueeze(0).to(device)

                sv = explainer.shap_values(input_feat, check_additivity=False)
                if isinstance(sv, list):
                    sv = sv[0]
                if torch.is_tensor(sv):
                    sv = sv.cpu().numpy()
                batch_shap.append(sv[0])  # (n_features,)

            inputs_s = np.stack([test_ds[i]["orf_features"][orf_index].numpy()
                                 for i in batch_idx])
            labels_s = np.array([test_ds[i]["label"].item() for i in batch_idx])

            all_shap_s.append(np.stack(batch_shap))
            all_inputs_s.append(inputs_s)
            all_labels_s.append(labels_s)

            print(f"  Explained {end}/{len(struct_explain_idx)} samples")

        shap_struct = np.concatenate(all_shap_s, axis=0)   # (N, n_features)
        input_struct = np.concatenate(all_inputs_s, axis=0)
        label_struct = np.concatenate(all_labels_s, axis=0)

        out_path = results_dir / f"deepshap_structural_{tag}{orf_suffix}{run_suffix}.npz"
        np.savez_compressed(out_path,
                            shap_values=shap_struct,
                            inputs=input_struct,
                            labels=label_struct,
                            explain_indices=struct_explain_idx,
                            feature_names=orf_feat_names,
                            seed=np.array(seed))
        print(f"  -> Saved {out_path} ({shap_struct.shape})")

    # --- Joint branch ---
    if not run_joint:
        pass  # skip
    else:
        n_ch = test_ds[0]["atg_windows"].shape[1]  # 9
        w_atg_actual = test_ds[0]["atg_windows"].shape[2]
        w_stop_actual = test_ds[0]["stop_windows"].shape[2]
        n_feat = test_ds[0]["orf_features"].shape[1]  # 5
        joint_dim = n_ch * w_atg_actual + n_ch * w_stop_actual + n_feat

        print(f"\n{'='*40}")
        print(f"DeepSHAP JOINT (ORF rank {orf_index} inputs, dim={joint_dim})")
        print(f"  ATG: {n_ch}x{w_atg_actual}={n_ch*w_atg_actual}, "
              f"Stop: {n_ch}x{w_stop_actual}={n_ch*w_stop_actual}, "
              f"Struct: {n_feat}")
        print(f"{'='*40}")

        # Background: concatenated ORF inputs from training samples
        # Only use training samples where this ORF rank is present (mask=True)
        bg_parts = []
        for i in bg_idx:
            s = train_ds[i]
            if not s["orf_mask"][orf_index]:
                continue
            atg_flat = s["atg_windows"][orf_index].reshape(-1)    # (C*W_atg,)
            stop_flat = s["stop_windows"][orf_index].reshape(-1)  # (C*W_stop,)
            struct_flat = s["orf_features"][orf_index]             # (F,)
            bg_parts.append(torch.cat([atg_flat, stop_flat, struct_flat]))
        if len(bg_parts) < 10:
            print(f"  WARNING: only {len(bg_parts)} background samples have ORF rank {orf_index}")
        bg_joint = torch.stack(bg_parts).float().to(device)  # (n_bg, joint_dim)
        print(f"  Background tensor: {bg_joint.shape}")

        all_shap_j = []
        all_inputs_j = []
        all_labels_j = []

        # Filter to samples where this ORF rank is present
        joint_explain_idx = np.array([i for i in explain_idx
                                      if test_ds[i]["orf_mask"][orf_index]])
        if len(joint_explain_idx) < len(explain_idx):
            print(f"  Filtered to {len(joint_explain_idx)}/{len(explain_idx)} "
                  f"samples with ORF rank {orf_index} present")

        for start in range(0, len(joint_explain_idx), batch_size):
            end = min(start + batch_size, len(joint_explain_idx))
            batch_idx = joint_explain_idx[start:end]

            batch_shap = []
            for i in batch_idx:
                sample = test_ds[i]

                wrapper = JointBranchWrapper(
                    model, n_ch, w_atg_actual, w_stop_actual, n_feat,
                    fixed_atg_windows=sample["atg_windows"].unsqueeze(0).to(device),
                    fixed_stop_windows=sample["stop_windows"].unsqueeze(0).to(device),
                    fixed_orf_features=sample["orf_features"].unsqueeze(0).to(device),
                    fixed_orf_mask=sample["orf_mask"].unsqueeze(0).to(device),
                    orf_index=orf_index,
                )

                explainer = shap.DeepExplainer(wrapper, bg_joint)

                inp_flat = torch.cat([
                    sample["atg_windows"][orf_index].reshape(-1),
                    sample["stop_windows"][orf_index].reshape(-1),
                    sample["orf_features"][orf_index],
                ]).unsqueeze(0).float().to(device)

                sv = explainer.shap_values(inp_flat, check_additivity=False)
                if isinstance(sv, list):
                    sv = sv[0]
                if torch.is_tensor(sv):
                    sv = sv.cpu().numpy()
                batch_shap.append(sv[0])  # (joint_dim,)

            # Collect inputs and labels for this batch
            batch_inputs = []
            for i in batch_idx:
                s = test_ds[i]
                inp = torch.cat([
                    s["atg_windows"][orf_index].reshape(-1),
                    s["stop_windows"][orf_index].reshape(-1),
                    s["orf_features"][orf_index],
                ]).numpy()
                batch_inputs.append(inp)
            labels_j = np.array([test_ds[i]["label"].item() for i in batch_idx])

            all_shap_j.append(np.stack(batch_shap))
            all_inputs_j.append(np.stack(batch_inputs))
            all_labels_j.append(labels_j)

            print(f"  Explained {end}/{len(joint_explain_idx)} samples")

        shap_joint = np.concatenate(all_shap_j, axis=0)    # (N, joint_dim)
        input_joint = np.concatenate(all_inputs_j, axis=0)
        label_joint = np.concatenate(all_labels_j, axis=0)

        # Save with metadata for unpacking
        out_path = results_dir / f"deepshap_joint_{tag}{orf_suffix}{run_suffix}.npz"
        np.savez_compressed(out_path,
                            shap_values=shap_joint,
                            inputs=input_joint,
                            labels=label_joint,
                            explain_indices=joint_explain_idx,
                            n_channels=np.array(n_ch),
                            w_atg=np.array(w_atg_actual),
                            w_stop=np.array(w_stop_actual),
                            n_features=np.array(n_feat),
                            feature_names=orf_feat_names,
                            channel_names=["A", "C", "G", "T", "junction",
                                          "rolling_gc", "frame_0", "frame_1",
                                          "frame_2"],
                            seed=np.array(seed))
        print(f"  -> Saved {out_path} ({shap_joint.shape})")

        # Quick additivity check on first 10 NMD samples
        print("\n  Additivity check (first 10 NMD samples):")
        nmd_mask = label_joint == 1
        nmd_idx_check = np.where(nmd_mask)[0][:10]
        for idx in nmd_idx_check:
            shap_sum = shap_joint[idx].sum()
            sample = test_ds[joint_explain_idx[idx]]
            # Reconstruct this sample's wrapper for correct E[f(x)]
            check_wrapper = JointBranchWrapper(
                model, n_ch, w_atg_actual, w_stop_actual, n_feat,
                fixed_atg_windows=sample["atg_windows"].unsqueeze(0).to(device),
                fixed_stop_windows=sample["stop_windows"].unsqueeze(0).to(device),
                fixed_orf_features=sample["orf_features"].unsqueeze(0).to(device),
                fixed_orf_mask=sample["orf_mask"].unsqueeze(0).to(device),
                orf_index=orf_index,
            )
            with torch.no_grad():
                # f(x)
                inp_flat = torch.tensor(input_joint[idx:idx+1],
                                        dtype=torch.float32).to(device)
                pred = check_wrapper(inp_flat).item()
                # E[f(x)] from background through this sample's wrapper
                bg_preds = []
                for bi in range(0, len(bg_joint), 50):
                    bg_batch = bg_joint[bi:bi+50]
                    bg_out = check_wrapper(bg_batch)
                    bg_preds.append(bg_out.cpu().numpy())
                expected = np.concatenate(bg_preds).mean()
            residual = pred - expected - shap_sum
            print(f"    Sample {idx}: f(x)={pred:.4f}, E[f]={expected:.4f}, "
                  f"sum(SHAP)={shap_sum:.4f}, residual={residual:.4f}")

    # Summary statistics
    print("\nComputing summary statistics ...")
    summary_rows = []

    # Sequence branches (ATG, stop): per-channel, averaged across positions
    for branch in seq_branches:
        npz_path = results_dir / f"deepshap_{branch}_{tag}{orf_suffix}{run_suffix}.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        shap_vals = data["shap_values"]
        labels = data["labels"]
        channel_names = data["channel_names"]

        for ch_idx, ch_name in enumerate(channel_names):
            mean_abs = np.abs(shap_vals[:, ch_idx, :]).mean()
            mean_abs_nmd = np.abs(shap_vals[labels == 1, ch_idx, :]).mean() if labels.sum() > 0 else 0
            mean_abs_ctrl = np.abs(shap_vals[labels == 0, ch_idx, :]).mean() if (1-labels).sum() > 0 else 0
            summary_rows.append({
                "branch": branch,
                "channel": ch_name,
                "mean_abs_shap": mean_abs,
                "mean_abs_shap_nmd": mean_abs_nmd,
                "mean_abs_shap_ctrl": mean_abs_ctrl,
            })

    # Structural branch: per-feature (no positional dimension)
    struct_npz = results_dir / f"deepshap_structural_{tag}{orf_suffix}{run_suffix}.npz"
    if run_structural and struct_npz.exists():
        struct_data = np.load(struct_npz)
        struct_shap = struct_data["shap_values"]   # (N, n_features) or (N, n_features, 1)
        if struct_shap.ndim == 3:
            struct_shap = struct_shap.squeeze(-1)
        struct_labels = struct_data["labels"]
        struct_feat_names = list(struct_data["feature_names"])

        for f_idx, f_name in enumerate(struct_feat_names):
            mean_abs = np.abs(struct_shap[:, f_idx]).mean()
            mean_abs_nmd = np.abs(struct_shap[struct_labels == 1, f_idx]).mean() if struct_labels.sum() > 0 else 0
            mean_abs_ctrl = np.abs(struct_shap[struct_labels == 0, f_idx]).mean() if (1-struct_labels).sum() > 0 else 0
            summary_rows.append({
                "branch": "structural",
                "channel": f_name,
                "mean_abs_shap": mean_abs,
                "mean_abs_shap_nmd": mean_abs_nmd,
                "mean_abs_shap_ctrl": mean_abs_ctrl,
            })

    # Joint branch: summarize by splitting back into ATG/stop/structural regions
    joint_npz = results_dir / f"deepshap_joint_{tag}{orf_suffix}{run_suffix}.npz"
    if run_joint and joint_npz.exists():
        jd = np.load(joint_npz)
        jshap = jd["shap_values"]
        if jshap.ndim == 3:
            jshap = jshap.squeeze(-1)
        jlabels = jd["labels"]
        j_n_ch = int(jd["n_channels"])
        j_w_atg = int(jd["w_atg"])
        j_w_stop = int(jd["w_stop"])
        j_n_feat = int(jd["n_features"])
        j_ch_names = list(jd["channel_names"])
        j_feat_names = list(jd["feature_names"])

        atg_end = j_n_ch * j_w_atg
        stop_end = atg_end + j_n_ch * j_w_stop

        # ATG region: reshape to (N, C, W_atg), summarize per channel
        atg_shap = jshap[:, :atg_end].reshape(-1, j_n_ch, j_w_atg)
        for ci, ch in enumerate(j_ch_names):
            # Per-position mean (for compatibility with marginal summary)
            mean_abs = np.abs(atg_shap[:, ci, :]).mean()
            mean_abs_nmd = np.abs(atg_shap[jlabels == 1, ci, :]).mean() if jlabels.sum() > 0 else 0
            mean_abs_ctrl = np.abs(atg_shap[jlabels == 0, ci, :]).mean() if (1 - jlabels).sum() > 0 else 0
            # Position-summed total per sample
            total_nmd = np.abs(atg_shap[jlabels == 1, ci, :]).sum(axis=1).mean() if jlabels.sum() > 0 else 0
            total_ctrl = np.abs(atg_shap[jlabels == 0, ci, :]).sum(axis=1).mean() if (1 - jlabels).sum() > 0 else 0
            summary_rows.append({
                "branch": "joint_atg", "channel": ch,
                "mean_abs_shap": mean_abs, "mean_abs_shap_nmd": mean_abs_nmd,
                "mean_abs_shap_ctrl": mean_abs_ctrl,
                "total_abs_shap_nmd": total_nmd, "total_abs_shap_ctrl": total_ctrl,
            })

        # Stop region
        stop_shap = jshap[:, atg_end:stop_end].reshape(-1, j_n_ch, j_w_stop)
        for ci, ch in enumerate(j_ch_names):
            mean_abs = np.abs(stop_shap[:, ci, :]).mean()
            mean_abs_nmd = np.abs(stop_shap[jlabels == 1, ci, :]).mean() if jlabels.sum() > 0 else 0
            mean_abs_ctrl = np.abs(stop_shap[jlabels == 0, ci, :]).mean() if (1 - jlabels).sum() > 0 else 0
            total_nmd = np.abs(stop_shap[jlabels == 1, ci, :]).sum(axis=1).mean() if jlabels.sum() > 0 else 0
            total_ctrl = np.abs(stop_shap[jlabels == 0, ci, :]).sum(axis=1).mean() if (1 - jlabels).sum() > 0 else 0
            summary_rows.append({
                "branch": "joint_stop", "channel": ch,
                "mean_abs_shap": mean_abs, "mean_abs_shap_nmd": mean_abs_nmd,
                "mean_abs_shap_ctrl": mean_abs_ctrl,
                "total_abs_shap_nmd": total_nmd, "total_abs_shap_ctrl": total_ctrl,
            })

        # Structural region
        struct_shap_j = jshap[:, stop_end:]
        for fi, fn in enumerate(j_feat_names):
            mean_abs = np.abs(struct_shap_j[:, fi]).mean()
            mean_abs_nmd = np.abs(struct_shap_j[jlabels == 1, fi]).mean() if jlabels.sum() > 0 else 0
            mean_abs_ctrl = np.abs(struct_shap_j[jlabels == 0, fi]).mean() if (1 - jlabels).sum() > 0 else 0
            summary_rows.append({
                "branch": "joint_structural", "channel": fn,
                "mean_abs_shap": mean_abs, "mean_abs_shap_nmd": mean_abs_nmd,
                "mean_abs_shap_ctrl": mean_abs_ctrl,
                "total_abs_shap_nmd": mean_abs_nmd,  # same (no positional dim)
                "total_abs_shap_ctrl": mean_abs_ctrl,
            })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = results_dir / f"deepshap_summary_{tag}{orf_suffix}{run_suffix}.tsv"
        summary_df.to_csv(summary_path, sep="\t", index=False)
        print(f"  -> {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n-explain", type=int, default=2000)
    parser.add_argument("--n-background", type=int, default=100)
    parser.add_argument("--atg-window", type=int, default=None,
                        help="Override window_size_atg from config")
    parser.add_argument("--stop-window", type=int, default=None,
                        help="Override window_size_stop from config")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed (default: from config)")
    parser.add_argument("--run-id", type=int, default=None,
                        help="Run identifier for replicate experiments")
    parser.add_argument("--branches", nargs="+", default=None,
                        choices=["atg", "stop", "structural", "joint"],
                        help="Which branches to run (default: atg stop structural)")
    parser.add_argument("--orf-index", type=int, default=0,
                        help="Which ORF rank to explain (0-4, default: 0)")
    args = parser.parse_args()
    run_deepshap(args.config, args.n_explain, args.n_background,
                 args.atg_window, args.stop_window, args.seed, args.run_id,
                 args.branches, args.orf_index)

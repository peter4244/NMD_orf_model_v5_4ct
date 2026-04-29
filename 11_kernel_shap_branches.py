#!/usr/bin/env python3
"""
11_kernel_shap_branches.py — Exact Shapley branch decomposition via KernelSHAP.

Decomposes the model prediction into three additive branch contributions:
  φ_ATG + φ_stop + φ_structural = f(x) - E[f(x)]

Uses embedding-level intervention: pre-computes the 32-dim sub-embeddings
from each branch, then evaluates all 2^3 = 8 coalitions to compute exact
Shapley values for 3 "players" (ATG branch, stop branch, structural branch).

For missing branches in a coalition, integrates over background sample
embeddings (not mean approximation) to correctly handle ReLU nonlinearity.
"""

import argparse
import json
import math
from pathlib import Path
from itertools import product

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import build_model
from utils import NMDDataset, load_config, set_seed


def extract_sub_embeddings(model, dataset, indices, device, batch_size=256):
    """
    Extract the three 32-dim sub-embeddings for rank-0 ORF of each sample.
    Also returns the full ORF embeddings for ranks 1-4 (fixed context),
    the orf_mask, and labels.
    """
    orf_index = 0
    encoder = model.orf_encoder

    all_atg_emb = []
    all_stop_emb = []
    all_struct_emb = []
    all_other_orf_emb = []  # (N, K-1, 64) for ranks 1-4
    all_masks = []
    all_labels = []

    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        batch_idx = indices[start:end]

        atg_wins = []
        stop_wins = []
        orf_feats = []
        full_atg = []
        full_stop = []
        full_feat = []
        masks = []
        labels = []

        for i in batch_idx:
            s = dataset[i]
            atg_wins.append(s["atg_windows"][orf_index])
            stop_wins.append(s["stop_windows"][orf_index])
            orf_feats.append(s["orf_features"][orf_index])
            full_atg.append(s["atg_windows"])
            full_stop.append(s["stop_windows"])
            full_feat.append(s["orf_features"])
            masks.append(s["orf_mask"])
            labels.append(s["label"].item())

        # Rank-0 sub-embeddings
        atg_batch = torch.stack(atg_wins).to(device)
        stop_batch = torch.stack(stop_wins).to(device)
        feat_batch = torch.stack(orf_feats).to(device)

        with torch.no_grad():
            atg_emb = encoder.atg_cnn(atg_batch)              # (B, 32)
            stop_emb = encoder.stop_cnn(stop_batch)            # (B, 32)
            struct_emb = F.relu(encoder.struct_fc(feat_batch))  # (B, 32)

        all_atg_emb.append(atg_emb.cpu())
        all_stop_emb.append(stop_emb.cpu())
        all_struct_emb.append(struct_emb.cpu())

        # Full ORF embeddings for ranks 1-4 (needed as fixed context)
        full_atg_t = torch.stack(full_atg).to(device)    # (B, K, C, W)
        full_stop_t = torch.stack(full_stop).to(device)
        full_feat_t = torch.stack(full_feat).to(device)
        mask_t = torch.stack(masks).to(device)

        B, K = mask_t.shape
        embed_dim = model.aggregator.attn_score.in_features

        # Compute all ORF embeddings (same logic as model.forward)
        flat_mask = mask_t.reshape(-1)
        atg_flat = full_atg_t.reshape(B * K, *full_atg_t.shape[2:])
        stop_flat = full_stop_t.reshape(B * K, *full_stop_t.shape[2:])
        feat_flat = full_feat_t.reshape(B * K, -1)

        orf_emb_all = torch.zeros(B * K, embed_dim, device=device)
        valid = flat_mask.bool()
        if valid.sum() > 0:
            with torch.no_grad():
                valid_emb = encoder(atg_flat[valid], stop_flat[valid], feat_flat[valid])
            orf_emb_all[valid] = valid_emb

        orf_emb_all = orf_emb_all.reshape(B, K, embed_dim)

        # Extract ranks 1-4
        other_emb = torch.cat([orf_emb_all[:, :orf_index],
                               orf_emb_all[:, orf_index+1:]], dim=1)  # (B, K-1, 64)
        all_other_orf_emb.append(other_emb.cpu())
        all_masks.append(mask_t.cpu())
        all_labels.extend(labels)

        if end % 2000 == 0 or end == len(indices):
            print(f"  Extracted {end}/{len(indices)} embeddings")

    return {
        "atg_emb": torch.cat(all_atg_emb),       # (N, 32)
        "stop_emb": torch.cat(all_stop_emb),      # (N, 32)
        "struct_emb": torch.cat(all_struct_emb),   # (N, 32)
        "other_orf_emb": torch.cat(all_other_orf_emb),  # (N, K-1, 64)
        "masks": torch.cat(all_masks),             # (N, K)
        "labels": np.array(all_labels),
    }


def evaluate_coalition(model, coalition, obs_embs, bg_embs, other_orf_emb,
                       orf_mask, device, orf_index=0):
    """
    Evaluate the model for a given coalition of branches.

    coalition: tuple of 3 bools (atg_present, stop_present, struct_present)
    obs_embs: dict with 'atg', 'stop', 'struct' tensors for this sample (each 32-dim)
    bg_embs: dict with 'atg', 'stop', 'struct' tensors for all bg samples (each N_bg × 32)
    other_orf_emb: (K-1, 64) fixed context for non-rank-0 ORFs
    orf_mask: (K,) mask for this sample

    Returns: scalar — mean f(x) over background integration for missing branches.
    """
    encoder = model.orf_encoder
    n_bg = bg_embs["atg"].shape[0]
    K = orf_mask.shape[0]
    embed_dim = model.aggregator.attn_score.in_features

    # Determine which branches need background integration
    missing = [i for i, present in enumerate(coalition) if not present]
    branch_keys = ["atg", "stop", "struct"]

    if len(missing) == 0:
        # All present: single forward pass
        concat = torch.cat([obs_embs["atg"], obs_embs["stop"], obs_embs["struct"]],
                           dim=-1).unsqueeze(0)  # (1, 96)
        with torch.no_grad():
            rank0_emb = encoder.fusion(concat)  # (1, 64)

            # Assemble K ORF embeddings
            orf_embs = torch.zeros(1, K, embed_dim, device=device)
            orf_embs[0, :orf_index] = other_orf_emb[:orf_index]
            orf_embs[0, orf_index] = rank0_emb[0]
            orf_embs[0, orf_index+1:] = other_orf_emb[orf_index:]

            mask = orf_mask.unsqueeze(0)  # (1, K)
            tx_emb, _ = model.aggregator(orf_embs, mask)
            logit = model.cls_head(model.head(tx_emb))

        return logit.item()

    # Some branches missing: integrate over background
    # For efficiency, batch all bg combinations
    # Present branches: tile to n_bg copies
    # Missing branches: use each bg sample's embedding

    parts = []
    for i, key in enumerate(branch_keys):
        if coalition[i]:
            parts.append(obs_embs[key].unsqueeze(0).expand(n_bg, -1))  # (n_bg, 32)
        else:
            parts.append(bg_embs[key])  # (n_bg, 32)

    concat = torch.cat(parts, dim=-1)  # (n_bg, 96)

    with torch.no_grad():
        rank0_emb = encoder.fusion(concat)  # (n_bg, 64)

        orf_embs = torch.zeros(n_bg, K, embed_dim, device=device)
        orf_embs[:, :orf_index] = other_orf_emb[:orf_index].unsqueeze(0).expand(n_bg, -1, -1)
        orf_embs[:, orf_index] = rank0_emb
        orf_embs[:, orf_index+1:] = other_orf_emb[orf_index:].unsqueeze(0).expand(n_bg, -1, -1)

        mask = orf_mask.unsqueeze(0).expand(n_bg, -1)
        tx_emb, _ = model.aggregator(orf_embs, mask)
        logits = model.cls_head(model.head(tx_emb))  # (n_bg, 1)

    return logits.mean().item()


def compute_shapley_3(values):
    """
    Exact Shapley values for 3 players from all 2^3 coalition values.

    values: dict mapping coalition tuple → v(S)
      e.g., (False, False, False) → v({}), (True, False, True) → v({1,3})

    Returns: (φ_1, φ_2, φ_3)
    """
    v = values

    # Shapley formula for n=3:
    # φ_i = Σ_{S⊆N\{i}} [|S|!(n-|S|-1)!/n!] × [v(S∪{i}) - v(S)]
    # Weights: |S|=0 → 2/6=1/3, |S|=1 → 1/6, |S|=2 → 2/6=1/3

    def phi(i):
        """Shapley value for player i (0-indexed)."""
        total = 0.0
        others = [j for j in range(3) if j != i]

        # All subsets of others (2^2 = 4 subsets)
        for bits in range(4):
            S = [False, False, False]
            s_size = 0
            for k, j in enumerate(others):
                if bits & (1 << k):
                    S[j] = True
                    s_size += 1

            S_with_i = list(S)
            S_with_i[i] = True

            weight = (
                math.factorial(s_size) *
                math.factorial(3 - s_size - 1) /
                math.factorial(3)
            )

            marginal = v[tuple(S_with_i)] - v[tuple(S)]
            total += weight * marginal

        return total

    return phi(0), phi(1), phi(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--tag", default="atg500_stop500")
    parser.add_argument("--n-background", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ws_atg = int(args.tag.split("_")[0].replace("atg", ""))
    ws_stop = int(args.tag.split("_")[1].replace("stop", ""))
    results_dir = Path("results")
    h5_path = config["data"]["hdf5_path"]

    print(f"=== KernelSHAP Branch Decomposition ===")
    print(f"Tag: {args.tag}, Background: {args.n_background}, Device: {device}")

    # Load model
    ckpt_path = results_dir / f"best_model_{args.tag}.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_config = {**config["model"],
                    "window_size_atg": ws_atg, "window_size_stop": ws_stop}
    model = build_model(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded from {ckpt_path}")

    # Load data
    test_ds = NMDDataset(h5_path, ws_atg, ws_stop, split="test")
    train_ds = NMDDataset(h5_path, ws_atg, ws_stop, split="train")
    print(f"Test: {len(test_ds)}, Train: {len(train_ds)}")

    # Load isoform IDs
    with h5py.File(h5_path, 'r') as f:
        all_ids = np.array([x.decode() if isinstance(x, bytes) else x
                            for x in f['isoform_id'][:]])
    test_ids = all_ids[test_ds.indices]

    # Select background
    rng = np.random.RandomState(args.seed)
    bg_idx = rng.choice(len(train_ds), size=min(args.n_background, len(train_ds)),
                        replace=False)

    # Pre-compute embeddings
    print("\nPre-computing test embeddings ...")
    test_embs = extract_sub_embeddings(model, test_ds, np.arange(len(test_ds)),
                                        device)
    print(f"  Test: atg={test_embs['atg_emb'].shape}, labels={test_embs['labels'].shape}")

    print("Pre-computing background embeddings ...")
    bg_embs_raw = extract_sub_embeddings(model, train_ds, bg_idx, device)
    bg_embs = {
        "atg": bg_embs_raw["atg_emb"].to(device),     # (N_bg, 32)
        "stop": bg_embs_raw["stop_emb"].to(device),
        "struct": bg_embs_raw["struct_emb"].to(device),
    }
    print(f"  Background: {bg_embs['atg'].shape[0]} samples")

    # All 8 coalitions
    coalitions = list(product([False, True], repeat=3))

    # Compute Shapley values for all test samples
    print(f"\nComputing Shapley values for {len(test_ds)} samples ...")
    results = []
    n_test = len(test_ds)

    for idx in range(n_test):
        obs = {
            "atg": test_embs["atg_emb"][idx].to(device),
            "stop": test_embs["stop_emb"][idx].to(device),
            "struct": test_embs["struct_emb"][idx].to(device),
        }
        other_emb = test_embs["other_orf_emb"][idx].to(device)
        mask = test_embs["masks"][idx].to(device)

        # Evaluate all 8 coalitions
        v = {}
        for coal in coalitions:
            v[coal] = evaluate_coalition(model, coal, obs, bg_embs,
                                         other_emb, mask, device)

        # Exact Shapley values
        phi_atg, phi_stop, phi_struct = compute_shapley_3(v)

        # Additivity check
        fx = v[(True, True, True)]
        efx = v[(False, False, False)]
        shap_sum = phi_atg + phi_stop + phi_struct
        residual = fx - efx - shap_sum

        results.append({
            "isoform_id": test_ids[idx],
            "label": float(test_embs["labels"][idx]),
            "prediction": fx,
            "expected_value": efx,
            "shap_atg": phi_atg,
            "shap_stop": phi_stop,
            "shap_structural": phi_struct,
            "shap_sum": shap_sum,
            "residual": residual,
        })

        if (idx + 1) % 1000 == 0 or idx == n_test - 1:
            print(f"  {idx+1}/{n_test} (last residual: {residual:.6f})")

    df = pd.DataFrame(results)

    # Summary
    print(f"\nAdditivity check:")
    print(f"  Mean |residual|: {df['residual'].abs().mean():.8f}")
    print(f"  Max |residual|:  {df['residual'].abs().max():.8f}")
    print(f"  (Should be ~0 for exact Shapley values)")

    nmd = df[df.label > 0.5]
    ctrl = df[df.label < 0.5]
    print(f"\nBranch importance (mean |SHAP|):")
    for branch in ["shap_atg", "shap_stop", "shap_structural"]:
        nmd_val = nmd[branch].abs().mean()
        ctrl_val = ctrl[branch].abs().mean()
        print(f"  {branch:20s}: NMD={nmd_val:.4f}, Control={ctrl_val:.4f}")

    total_nmd = nmd[["shap_atg", "shap_stop", "shap_structural"]].abs().sum(axis=1).mean()
    print(f"\nBranch % of total |SHAP| (NMD):")
    for branch in ["shap_atg", "shap_stop", "shap_structural"]:
        pct = 100 * nmd[branch].abs().mean() / total_nmd
        print(f"  {branch:20s}: {pct:.1f}%")

    # Save
    out_path = results_dir / f"kernel_shap_branch_{args.tag}.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\n-> {out_path} ({len(df)} rows)")
    print("Done.")


if __name__ == "__main__":
    main()

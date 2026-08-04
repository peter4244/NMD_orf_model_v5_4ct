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
import gc
import math
import sys
from pathlib import Path
from itertools import product

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model import build_model
from evaluate import enforce_split_gate
from utils import (NMDDataset, load_config, member_tag, resolve_checkpoint, run_suffix,
                   selected_tag, set_seed, split_indices)


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

        if end % (batch_size * 8) == 0 or end == len(indices):
            print(f"  Extracted {end}/{len(indices)} embeddings")

    return {
        "atg_emb": torch.cat(all_atg_emb),       # (N, 32)
        "stop_emb": torch.cat(all_stop_emb),      # (N, 32)
        "struct_emb": torch.cat(all_struct_emb),   # (N, 32)
        "other_orf_emb": torch.cat(all_other_orf_emb),  # (N, K-1, 64)
        "masks": torch.cat(all_masks),             # (N, K)
        "labels": np.array(all_labels),
    }


def chunk_bounds(n, chunk_size, batch_size=256):
    """The chunk decomposition, as ONE function, so a test can drive THIS and not a copy of it.

    It lived inline in extract_sub_embeddings_chunked, which meant test_guards could only
    re-implement the loop bound and check its own copy -- and a copy of the logic is exactly the
    root cause this file's own docstrings keep naming. An off-by-one here now fails the guard.

    Returns [(lo, hi)] half-open ranges of split positions, ascending, covering 0..n exactly once.
    chunk_size must be a multiple of batch_size: every chunk boundary is then a batch boundary, so
    each forward pass receives the same rows in the same batch shape as an unchunked run -- and
    therefore selects the same library kernels, which is what keeps the two bit-comparable.
    """
    if chunk_size is None or chunk_size <= 0 or chunk_size >= n:
        return [(0, n)]
    if chunk_size % batch_size:
        raise ValueError(
            f"--chunk-size {chunk_size} is not a multiple of the extraction batch size "
            f"{batch_size}. A chunk boundary that is not a batch boundary re-partitions the "
            f"forward passes into different batch shapes, which can select different library "
            f"kernels and makes this run non-comparable with an unchunked one at the last bits. "
            f"Use a multiple of {batch_size}.")
    return [(lo, min(lo + chunk_size, n)) for lo in range(0, n, chunk_size)]


def extract_sub_embeddings_chunked(model, h5_path, ws_atg, ws_stop, split, device,
                                   chunk_size, batch_size=256):
    """extract_sub_embeddings over a split, reading the windows a chunk of rows at a time.

    WHY (2026-07-30, measured). NMDDataset materialises a split's windows as float32 on
    construction: 14.00 GiB for the full cohort at atg1000_stop1000, 28.01 GiB at
    atg2000_stop2000. And because `f[...][idx].astype(np.float32)` holds the float16 read
    buffer and the float32 result at the same time, the TRANSIENT peak is half an array higher
    again -- 35.3 GiB at the wider configuration, against the 32 GB slurm_kernel_shap_dn.sh
    asks for. Measured, not derived: at n=6,000/w2000 the model predicts 5.31 GiB and the
    process peaked at 5.38.

    None of it is needed. This function keeps only the three 32-dim branch embeddings, the
    ranks 1-4 context and the mask -- about 59 MB for all 41,765 isoforms. The windows are read
    once, consumed once and dropped. Peak falls to 2.10 GiB at atg1000_stop1000 and 3.66 GiB at
    atg2000_stop2000 -- it is NOT width-independent -- which is what puts a full-cohort run back
    on a laptop, where the plan's working loop assumes development happens. Measured on Explorer
    2026-07-31, largest sacct MaxRSS over the 25 cells of each configuration, array job 8850292.

    IT CHANGES NO NUMBER. Chunks are contiguous ascending ranges of split positions, so
    restrict_to's sort is the identity and concatenation restores the split's own row order
    exactly. chunk_size is a multiple of batch_size, so every chunk boundary is a batch boundary
    and each forward pass receives the same rows in the same batch SHAPE as the unchunked run --
    which matters because batch shape selects the library kernel, not because of any
    cross-sample reduction: there is none on this path. The encoders are convolution, batch
    norm and a max over length, and the model is in eval mode, so a row's embedding never
    depends on its batch-mates. The model is built once, outside the loop. There is no RNG here.
    Measured: on `val` at atg1000_stop1000 the chunked and eager outputs are byte-identical --
    same md5, all 4,356 rows agreeing to exactly zero. See chunked_read_equivalence_runlog.txt.
    """
    idx = split_indices(h5_path, split)
    n = len(idx)
    bounds = chunk_bounds(n, chunk_size, batch_size)

    if len(bounds) == 1 and bounds[0] == (0, n) and (chunk_size is None or chunk_size <= 0
                                                     or chunk_size >= n):
        ds = NMDDataset(h5_path, ws_atg, ws_stop, split=split)
        return extract_sub_embeddings(model, ds, np.arange(len(ds)), device, batch_size)

    parts = []
    n_chunks = len(bounds)
    for c, (lo, hi) in enumerate(bounds, start=1):
        print(f"  chunk {c}/{n_chunks}: split rows {lo:,}-{hi - 1:,}")
        ds = NMDDataset(h5_path, ws_atg, ws_stop, split=split,
                        restrict_to=np.arange(lo, hi))
        parts.append(extract_sub_embeddings(model, ds, np.arange(len(ds)), device, batch_size))
        del ds
        gc.collect()

    out = {k: torch.cat([p[k] for p in parts])
           for k in ("atg_emb", "stop_emb", "struct_emb", "other_orf_emb", "masks")}
    out["labels"] = np.concatenate([p["labels"] for p in parts])

    # The split's row count is fixed before the loop; a short concatenation would mean a chunk
    # silently returned fewer rows than it was given, and every downstream row would be
    # misaligned against test_ids.
    if len(out["labels"]) != n:
        raise RuntimeError(f"chunked extraction returned {len(out['labels']):,} rows for a "
                           f"split of {n:,} -- rows were lost, and isoform_id alignment is void")
    return out


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
    # A LOG THAT ARRIVES WHILE THE JOB IS ALIVE (2026-07-30). Python block-buffers stdout when it
    # is not a terminal, so under SLURM every print below sat in a 4 KB buffer and the log stayed
    # empty for minutes. That makes a healthy run and a dead one look identical -- which is the
    # first of this script's recorded operational failures, "a job that died at 17 seconds having
    # produced no log". Set here rather than as `python -u` in the SLURM script, because the
    # property should hold however the script is invoked.
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    # Parameterised 2026-07-27 for the same reason as 03_train.py: hardcoded, this reads the
    # PUBLISHED checkpoint and overwrites the PUBLISHED kernel_shap TSV -- the artifacts the
    # deposit-native run exists to be compared against. Default unchanged.
    parser.add_argument("--results-dir", default="results_4ct",
                        help="where the checkpoint is read and the SHAP table written")
    parser.add_argument("--tag", default=None,
                        help="Window-config tag. Default: the `selected:` block in "
                             "--config. Never a hardcoded literal -- see selected_tag.")
    parser.add_argument("--n-background", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--explain-split", default="test",
                        help="isoforms to explain: 'test' (default), 'all' "
                             "(train+val+test+paralog, full cohort), 'test_all', 'train', 'val'. "
                             "Background is always drawn from train.")
    parser.add_argument("--member-seed", type=int, default=None, help="Ensemble member to load, by training seed. Omitted = the legacy un-seeded checkpoint. Never silently guesses a member; see utils.resolve_checkpoint.")
    # DISTINCT FROM --seed AND FROM --member-seed, deliberately, and this file is where the three
    # axes finally become separable:
    #   --seed         this run's sampling RNG          (held FIXED across everything)
    #   --member-seed  which trained checkpoint         -> between-member / TRAINING variance
    #   --run-id       which background draw            -> within-member / INTERPRETATION variance
    # Same split of responsibilities deepshap.py already carries. Omitted reproduces the previous
    # single-draw behaviour exactly -- see the bg RNG below and utils.run_suffix.
    parser.add_argument("--run-id", type=int, default=None,
                        help="Attribution replicate id. Varies the BACKGROUND draw only; the "
                             "checkpoint and the explained set are untouched. Omitted = one "
                             "unreplicated draw, byte-identical to the pre-2026-07-30 behaviour.")
    # READING A CHECKPOINT AND WRITING OUTPUTS ARE DIFFERENT DIRECTORIES (2026-07-30), the same
    # separation deepshap.py gained at a301ffe. resolve_checkpoint was reading from --results-dir,
    # so explaining the sweep's members meant writing 50 interpretation TSVs into the tree that
    # holds the trained models. Omitted = --results-dir, so every existing invocation is unchanged.
    parser.add_argument("--checkpoint-dir", default=None,
                        help="where trained members live, if not --results-dir. Outputs always go "
                             "to --results-dir.")
    # MEMORY, WHICH IS THE BINDING CONSTRAINT HERE AND NOT TIME (2026-07-30). Reading the
    # explained split's windows in chunks instead of all at once. Must be a multiple of the 256
    # extraction batch size so batch composition -- and therefore floating-point reduction order
    # -- is unchanged; see extract_sub_embeddings_chunked. 0 restores the single eager read.
    parser.add_argument("--chunk-size", type=int, default=2048,
                        help="explained-split rows held in memory at once (multiple of 256). "
                             "0 = read the whole split eagerly, the pre-2026-07-30 behaviour, "
                             "which peaks at 35.3 GiB on a full cohort at atg2000_stop2000.")
    # THE SPLIT GATE, IMPORTED NOT RESTATED (2026-07-30). This file scored any split with no
    # affirmation at all: `--explain-split test` was a final evaluation nothing marked as one, and
    # `--explain-split all` pooled train and held-out data with nothing saying so. That is the hole
    # C69 found in evaluate.py via `--split all` and closed at 420a264 -- still open here, in a
    # producer of four section-5 claims. D31 and claim 5.6.4 rest on the test set being touched
    # once, deliberately, and a second scoring entry point without the gate is how that stops being
    # true quietly.
    parser.add_argument("--final", action="store_true",
                        help="affirm a final, pre-registered evaluation on a test split")
    parser.add_argument("--full-cohort", action="store_true",
                        help="affirm a pooled train+test interpretation run (--explain-split all)")
    args = parser.parse_args()

    # An empty reference set makes evaluate_coalition average over zero rows -> NaN attributions,
    # a full run of them, and exit 0. Validated here rather than at first use, so a typo costs a
    # message instead of a checkpoint load and a window read.
    if args.n_background < 1:
        parser.error(f"--n-background {args.n_background}: the reference set would be empty and "
                     f"every attribution would be NaN at exit 0.")

    # evaluate.enforce_split_gate reads args.split; this file's flag is --explain-split for
    # continuity with the wrappers that already pass it. Aliasing here rather than renaming the
    # flag keeps slurm_kernel_shap*.sh working and still leaves ONE definition of the gate.
    args.split = args.explain_split
    enforce_split_gate(parser, args)

    config = load_config(args.config)
    if args.tag is None:
        args.tag = selected_tag(config)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ws_atg = int(args.tag.split("_")[0].replace("atg", ""))
    ws_stop = int(args.tag.split("_")[1].replace("stop", ""))
    results_dir = Path(args.results_dir)
    h5_path = config["data"]["hdf5_path"]

    print(f"=== KernelSHAP Branch Decomposition ===")
    print(f"Tag: {args.tag}, Background: {args.n_background}, Device: {device}")

    # Load model
    ckpt_path = resolve_checkpoint(args.checkpoint_dir or results_dir, args.tag, args.member_seed)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_config = {**config["model"],
                    "window_size_atg": ws_atg, "window_size_stop": ws_stop}
    model = build_model(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded from {ckpt_path}")

    # THE EXPLAINED SET IS RESOLVED AS ROW POSITIONS, NOT MATERIALISED (2026-07-30). Naming the
    # isoforms about to be explained used to require constructing the whole split -- up to 28.01
    # GiB of float32 windows -- to reach its `.indices`. split_indices reads the `split` array
    # only. The windows are read later, in chunks, by extract_sub_embeddings_chunked.
    expl_idx = split_indices(h5_path, args.explain_split)

    # THE REFERENCE SET IS 500 ROWS, NOT A SPLIT (2026-07-30). NMDDataset materialises a split's
    # windows as float32 on construction, so building all of `train` to use 500 of its 26,711 rows
    # cost 9.0 GB at atg1000_stop1000 and 17.9 GB at atg2000_stop2000 -- and with the explained
    # cohort alongside, 46 GB for one full-cohort run at the wider configuration, against a 32 GB
    # SLURM request. Measured 2026-07-30 by running this script, not by reading it.
    #
    # The row count comes from the `split` array directly so the draw can happen before any window
    # data is read. choice(n_train, ...) is the SAME call on the SAME RandomState as before, so the
    # positions drawn are identical -- see the equivalence check in analysis_plans/.
    n_train = len(split_indices(h5_path, "train"))

    # Load isoform IDs
    with h5py.File(h5_path, 'r') as f:
        all_ids = np.array([x.decode() if isinstance(x, bytes) else x
                            for x in f['isoform_id'][:]])
    test_ids = all_ids[expl_idx]

    # Select background.
    #
    # THE REPLICATE AXIS, ADDED 2026-07-30. This is the only stochastic step in the whole script:
    # the explained set is the WHOLE split (np.arange below, not a subsample), so unlike
    # deepshap.py there is no explained-subset RNG to keep separate and --seed needs no second
    # RandomState. Varying the background is therefore a clean interpretation-variance knob.
    #
    # `+ 1000 * run_id` matches deepshap.py:275 rather than inventing a second scheme, and
    # run_id=None gives RandomState(args.seed) -- the exact draw every existing artifact was
    # computed from, so nothing already deposited becomes unreproducible.
    bg_rng = np.random.RandomState(args.seed + 1000 * (args.run_id or 0))
    bg_pos = bg_rng.choice(n_train, size=min(args.n_background, n_train), replace=False)
    train_ds = NMDDataset(h5_path, ws_atg, ws_stop, split="train", restrict_to=bg_pos)
    print(f"Explain ({args.explain_split}): {len(expl_idx)}, "
          f"Reference: {len(train_ds)} of {n_train:,} train rows")

    # Pre-compute embeddings. The explained split is read in chunks; the reference set is 500
    # rows and is read whole.
    print(f"\nPre-computing test embeddings (chunk size {args.chunk_size or 'off — eager'}) ...")
    test_embs = extract_sub_embeddings_chunked(model, h5_path, ws_atg, ws_stop,
                                               args.explain_split, device, args.chunk_size)
    print(f"  Test: atg={test_embs['atg_emb'].shape}, labels={test_embs['labels'].shape}")
    if len(test_embs["labels"]) != len(expl_idx):
        raise RuntimeError(
            f"explained embeddings hold {len(test_embs['labels']):,} rows but the split has "
            f"{len(expl_idx):,}; test_ids would be misaligned against them")

    print("Pre-computing background embeddings ...")
    # restrict_to already selected the drawn rows, so every row of train_ds is a reference isoform.
    # They are held in sorted order; evaluate_coalition averages over them, so order carries no
    # meaning -- only the floating-point summation order differs, at machine epsilon.
    bg_embs_raw = extract_sub_embeddings(model, train_ds, np.arange(len(train_ds)), device)
    bg_embs = {
        "atg": bg_embs_raw["atg_emb"].to(device),     # (N_bg, 32)
        "stop": bg_embs_raw["stop_emb"].to(device),
        "struct": bg_embs_raw["struct_emb"].to(device),
    }
    print(f"  Background: {bg_embs['atg'].shape[0]} samples")

    # All 8 coalitions
    coalitions = list(product([False, True], repeat=3))

    # Compute Shapley values for all test samples
    print(f"\nComputing Shapley values for {len(expl_idx)} samples ...")
    results = []
    n_test = len(expl_idx)

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

    # Save (suffix non-test explained sets so the full-cohort file is distinct)
    suffix = "" if args.explain_split == "test" else f"_{args.explain_split}"
    # Member in the name, same reasoning as deepshap.py: --member-seed chose the checkpoint (:283)
    # and the output name did not say which. member_tag(tag, None) == tag, so existing runs are
    # unaffected. The explain-split suffix was already handled just above -- this file was half
    # self-documenting and is now consistent.
    # THE REPLICATE SLOT (2026-07-30). The member was in the name and the replicate was not, so
    # five background draws on one member wrote one file five times -- member_tag's own collision,
    # one axis over, in the producer of four section-5 claims. utils.run_suffix is the rule;
    # run_suffix(None) == "" keeps every deposited filename byte-identical. test_guards.py §4.
    out_path = (results_dir /
                f"kernel_shap_branch_{member_tag(args.tag, args.member_seed)}"
                f"{suffix}{run_suffix(args.run_id)}.tsv")
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\n-> {out_path} ({len(df)} rows)")
    print("Done.")


if __name__ == "__main__":
    main()

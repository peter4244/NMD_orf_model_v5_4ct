#!/usr/bin/env python3
"""
12_feature_shap_structural.py — exact Shapley decomposition over the FIVE structural features.

Implements steps 3 and 4 of ANALYSIS3_PLAN_2026-07-31.md. Sibling of
11_kernel_shap_branches.py, which plays the three-player game over BRANCHES; this plays the
five-player game over the structural branch's inputs, with the two sequence branches and ORFs
1-4 held at their observed values.

EXACT, NOT APPROXIMATE, WHICH IS WHY IT IS NOT deepshap.py. Five players enumerate to 32
coalitions, and the features enter through nn.Linear(5, 32) -- so the expensive CNN encodings
are computed once per isoform and reused across every coalition, exactly as in the sibling.
That keeps the residual a checksum: the five attributions plus the baseline sum to the
prediction as an algebraic consequence, so anything materially above rounding means the file
did not come from the calculation it claims to. An approximate method would forfeit that.

THIS IS A DIFFERENT GAME FROM ANALYSIS 2, NOT A SUBDIVISION OF IT. Shapley values are not
additive under regrouping: these five attributions do NOT sum to the structural branch's
attribution from the three-player game, and the two must never be composed. Concretely, this
game's total is v(all five) - v(none), which is the structural effect with the sequence
branches held PRESENT; Analysis 2's phi_structural is a weighted average of four marginal
contributions of which that is only one, entering at weight 1/3.

The baseline column is named `baseline_struct_reference`, NOT `expected_value`: in Analysis 2
that name means all three branches absent, here it means only the structural features absent.
Same word, different quantity, in two tables a reader will join.
"""

import argparse
import gc
import math
import sys
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model import build_model
from evaluate import enforce_split_gate
from utils import (NMDDataset, load_config, member_tag, resolve_checkpoint, run_suffix,
                   selected_tag, set_seed, split_indices)

FEATS = ["frac_start", "frac_stop", "is_ref_cds", "is_sqanti_cds", "n_downstream_ejc"]
NF = len(FEATS)


def extract_fixed(model, dataset, device, batch_size=256):
    """Everything that does NOT vary across coalitions, computed once.

    The two sequence-branch embeddings at ORF rank 0, the ranks 1-4 context, the mask, the
    label, and the NORMALIZED rank-0 feature vector itself -- the last because the coalitions
    vary the features and re-run struct_fc, rather than varying an embedding.
    """
    enc = model.orf_encoder
    out = {k: [] for k in ("start", "stop", "ctx", "mask", "x", "label")}
    n = len(dataset)
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        s = [dataset[i] for i in range(lo, hi)]
        atg = torch.stack([r["atg_windows"] for r in s]).to(device)
        stp = torch.stack([r["stop_windows"] for r in s]).to(device)
        ft = torch.stack([r["orf_features"] for r in s]).to(device)
        mk = torch.stack([r["orf_mask"] for r in s]).to(device)
        with torch.no_grad():
            out["start"].append(enc.atg_cnn(atg[:, 0]).cpu())
            out["stop"].append(enc.stop_cnn(stp[:, 0]).cpu())
            B, K = mk.shape
            d = model.aggregator.attn_score.in_features
            flat = mk.reshape(-1).bool()
            emb = torch.zeros(B * K, d, device=device)
            if flat.any():
                emb[flat] = enc(atg.reshape(B * K, *atg.shape[2:])[flat],
                                stp.reshape(B * K, *stp.shape[2:])[flat],
                                ft.reshape(B * K, -1)[flat])
            emb = emb.reshape(B, K, d)
            out["ctx"].append(torch.cat([emb[:, :0], emb[:, 1:]], dim=1).cpu())
        out["mask"].append(mk.cpu())
        out["x"].append(ft[:, 0].cpu())                 # normalized rank-0 features
        out["label"].extend(r["label"].item() for r in s)
        if hi % (batch_size * 8) == 0 or hi == n:
            print(f"  encoded {hi}/{n}", flush=True)
    r = {k: torch.cat(v) for k, v in out.items() if k != "label"}
    r["label"] = np.array(out["label"])
    return r


def extract_fixed_chunked(model, h5_path, wa, ws, split, device, chunk_size, batch_size=256):
    """extract_fixed over a split, a chunk of rows at a time. Same rationale and the same
    boundary rule as 11_kernel_shap_branches.chunk_bounds, which this imports rather than
    restates."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ks", Path(__file__).parent / "11_kernel_shap_branches.py")
    ks = importlib.util.module_from_spec(spec); spec.loader.exec_module(ks)

    n = len(split_indices(h5_path, split))
    bounds = ks.chunk_bounds(n, chunk_size, batch_size)
    parts = []
    for c, (lo, hi) in enumerate(bounds, start=1):
        print(f"  chunk {c}/{len(bounds)}: split rows {lo:,}-{hi - 1:,}", flush=True)
        ds = (NMDDataset(h5_path, wa, ws, split=split) if bounds == [(0, n)]
              else NMDDataset(h5_path, wa, ws, split=split, restrict_to=np.arange(lo, hi)))
        parts.append(extract_fixed(model, ds, device, batch_size))
        del ds; gc.collect()
    r = {k: torch.cat([p[k] for p in parts]) for k in ("start", "stop", "ctx", "mask", "x")}
    r["label"] = np.concatenate([p["label"] for p in parts])
    if len(r["label"]) != n:
        raise RuntimeError(f"chunked extraction returned {len(r['label']):,} of {n:,} rows")
    return r


SUBSETS = [frozenset(c) for k in range(NF + 1) for c in combinations(range(NF), k)]
_W = {s: math.factorial(len(s)) * math.factorial(NF - len(s) - 1) / math.factorial(NF)
      for s in {frozenset(c) for k in range(NF) for c in combinations(range(NF), k)}}


def shapley_from_subsets(v):
    """Exact Shapley values for NF players from all 2^NF coalition values.

    phi_i = sum over S subset of N\\{i} of |S|!(n-|S|-1)!/n! * [v(S+i) - v(S)]
    """
    phi = np.zeros(NF)
    for i in range(NF):
        for S in SUBSETS:
            if i in S:
                continue
            phi[i] += _W[S] * (v[S | {i}] - v[S])
    return phi


def main():
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--results-dir", default="results_4ct")
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--member-seed", type=int, default=None)
    p.add_argument("--run-id", type=int, default=None)
    p.add_argument("--n-background", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--explain-split", default="test")
    p.add_argument("--chunk-size", type=int, default=2048)
    p.add_argument("--limit", type=int, default=None,
                   help="explain only the first N isoforms of the split. A COST PROBE ONLY -- "
                        "it makes the output a non-analysis subset, so the filename is marked.")
    p.add_argument("--final", action="store_true")
    p.add_argument("--full-cohort", action="store_true")
    args = p.parse_args()

    if args.n_background < 1:
        p.error(f"--n-background {args.n_background}: the reference set would be empty and every "
                f"attribution would be NaN at exit 0.")
    args.split = args.explain_split
    enforce_split_gate(p, args)

    cfg = load_config(args.config)
    if args.tag is None:
        args.tag = selected_tag(cfg)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wa = int(args.tag.split("_")[0].replace("atg", ""))
    ws = int(args.tag.split("_")[1].replace("stop", ""))
    rd = Path(args.results_dir)
    h5 = cfg["data"]["hdf5_path"]

    print(f"=== Structural-feature Shapley (5 players, {len(SUBSETS)} coalitions) ===")
    print(f"Tag {args.tag}  member {args.member_seed}  draw {args.run_id}  device {device}")

    ck = resolve_checkpoint(args.checkpoint_dir or rd, args.tag, args.member_seed)
    model = build_model({**cfg["model"], "window_size_atg": wa, "window_size_stop": ws}).to(device)
    model.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["model_state_dict"])
    model.eval()
    print(f"Model loaded from {ck}")

    expl_idx = split_indices(h5, args.explain_split)
    n_train = len(split_indices(h5, "train"))
    with h5py.File(h5, "r") as f:
        all_ids = np.array([x.decode() if isinstance(x, bytes) else x for x in f["isoform_id"][:]])

    # The SAME draw as Analysis 2 -- same RandomState, same call -- so the two games are played
    # against identical baselines. See ANALYSIS3_PLAN step 2.
    bg_rng = np.random.RandomState(args.seed + 1000 * (args.run_id or 0))
    bg_pos = bg_rng.choice(n_train, size=min(args.n_background, n_train), replace=False)
    ref_ds = NMDDataset(h5, wa, ws, split="train", restrict_to=bg_pos)
    ref_x = torch.stack([ref_ds[i]["orf_features"][0] for i in range(len(ref_ds))]).to(device)
    n_bg = ref_x.shape[0]
    print(f"Explain ({args.explain_split}): {len(expl_idx)}   Reference: {n_bg} of {n_train:,}")

    print("\nEncoding what does not vary ...")
    E = extract_fixed_chunked(model, h5, wa, ws, args.explain_split, device, args.chunk_size)
    ids = all_ids[expl_idx]
    n_total = len(ids)
    if args.limit:
        n_total = min(args.limit, n_total)
        print(f"  --limit {args.limit}: explaining the first {n_total:,} isoforms (COST PROBE)")

    # Index pairs (coalition k, feature f) for every f PRESENT in coalition k. Built once: the
    # scatter below then writes the observed value into exactly those slots and leaves the rest
    # at the reference isoform's value.
    _pairs = [(k, f) for k, S in enumerate(SUBSETS) for f in sorted(S)]
    SEL_S = torch.tensor([k for k, _ in _pairs], device=device, dtype=torch.long)
    SEL_F = torch.tensor([f for _, f in _pairs], device=device, dtype=torch.long)

    enc = model.orf_encoder
    rows = []
    for i in range(n_total):
        start = E["start"][i].to(device); stop = E["stop"][i].to(device)
        ctx = E["ctx"][i].to(device); mask = E["mask"][i].to(device)
        x = E["x"][i].to(device)
        K = mask.shape[0]; d = model.aggregator.attn_score.in_features

        # ALL 32 COALITIONS IN ONE PASS. Evaluated one at a time this is 32 sequential forward
        # passes of 500 rows per isoform, and the per-isoform cost came out at ~80 ms against a
        # predicted ~12 -- the work is tiny matmuls, so it is launch- and Python-bound rather
        # than arithmetic-bound. Stacking them into a single (32*n_bg) batch is an implementation
        # choice that changes no value: every row is still one (coalition, reference) pair
        # evaluated independently, and the model is in eval mode so a row never depends on its
        # batch-mates. The mask selection below is built once, outside the isoform loop.
        with torch.no_grad():
            feat = ref_x.unsqueeze(0).repeat(len(SUBSETS), 1, 1)          # (32, n_bg, 5)
            feat[SEL_S, :, SEL_F] = x[SEL_F].unsqueeze(-1)                 # observed where present
            feat = feat.reshape(-1, NF)
            st = F.relu(enc.struct_fc(feat))
            fused = enc.fusion(torch.cat([start.expand(feat.shape[0], -1),
                                          stop.expand(feat.shape[0], -1), st], dim=-1))
            embs = torch.zeros(feat.shape[0], K, d, device=device)
            embs[:, 0] = fused
            embs[:, 1:] = ctx.unsqueeze(0).expand(feat.shape[0], -1, -1)
            tx, _ = model.aggregator(embs, mask.unsqueeze(0).expand(feat.shape[0], -1))
            logits = model.cls_head(model.head(tx)).reshape(len(SUBSETS), n_bg)
            means = logits.mean(dim=1).cpu().numpy()
        v = {S: float(means[k]) for k, S in enumerate(SUBSETS)}

        phi = shapley_from_subsets(v)
        full = v[frozenset(range(NF))]; base = v[frozenset()]
        rows.append(dict(isoform_id=ids[i], label=float(E["label"][i]),
                         prediction=full, baseline_struct_reference=base,
                         **{f"shap_{f}": phi[k] for k, f in enumerate(FEATS)},
                         shap_sum=phi.sum(), residual=full - base - phi.sum()))
        if (i + 1) % 500 == 0 or i == n_total - 1:
            print(f"  {i+1}/{n_total} (last residual: {rows[-1]['residual']:.3e})", flush=True)

    df = pd.DataFrame(rows)
    print(f"\nAdditivity: mean |residual| {df.residual.abs().mean():.3e}   "
          f"max |residual| {df.residual.abs().max():.3e}   (step 5 bar 1e-12)")
    nmd = df[df.label > 0.5]
    m = {f: nmd[f"shap_{f}"].abs().mean() for f in FEATS}
    tot = sum(m.values())
    print(f"\nFeature shares over {len(nmd):,} NMD-labelled isoforms:")
    for f in FEATS:
        print(f"  {f:20s} {100*m[f]/tot:5.1f}%   mean|phi| {m[f]:.4f}")

    suffix = "" if args.explain_split == "test" else f"_{args.explain_split}"
    lim = f"_first{n_total}" if args.limit else ""
    out = rd / (f"feature_shap_{member_tag(args.tag, args.member_seed)}"
                f"{suffix}{run_suffix(args.run_id)}{lim}.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(f"\n-> {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
09b_export_subgroup_profiles.py — Export per-isoform structural SHAP and
subgroup-stratified positional/motif SHAP profiles from joint DeepSHAP NPZs.

These TSVs feed Sections 9.6 (input importance by subgroup), 9.7 (start
signals by subgroup), and 9.8 (stop signals by subgroup) of the report.
Subgroup logic mirrors the report's Section 7.1 definitions.

Outputs (in --results-dir):
  sample_shap_structural_{tag}.tsv
  shap_profile_atg_subgroup_joint_{tag}.tsv
  shap_profile_stop_subgroup_joint_{tag}.tsv
  motif_logo_atg_subgroup_joint_{tag}.tsv
  motif_logo_stop_subgroup_joint_{tag}.tsv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paths_config import load_config, selected_tag


PTC_LOST_CATS = {"ref_atg_lost", "no_ref_isoform",
                 "not_atg_in_target", "no_stop_in_target"}
PTC_RETAINED_CATS = {"no_downstream_ejc", "truncated_no_ejc"}


def assign_subgroup(row):
    if row.label == 0:
        return "Control"
    cat = row.category
    td2 = row.td2_downstream_ejc
    if cat == "effectively_ptc":
        return "NMD PTC+"
    if cat in PTC_LOST_CATS and pd.notna(td2) and td2 > 0:
        return "NMD PTC+"
    if cat in PTC_RETAINED_CATS:
        return "NMD PTC-, ref ATG retained"
    if cat in PTC_LOST_CATS:
        return "NMD PTC-, ref ATG lost"
    return "NMD other"


def load_runs(results_dir, tag, n_runs):
    """Load all run NPZs, returning averaged shap and inputs as per-branch arrays."""
    n_ch, w_atg, w_stop, n_feat = 9, 500, 500, 5
    atg_size = n_ch * w_atg
    stop_size = n_ch * w_stop

    atg_shap_runs, stop_shap_runs, struct_shap_runs = [], [], []
    atg_in_runs, stop_in_runs = [], []
    labels = None
    for r in range(1, n_runs + 1):
        path = results_dir / f"deepshap_joint_{tag}_run{r}.npz"
        d = np.load(path, allow_pickle=True)
        sv = d["shap_values"].squeeze(-1)        # (N, 9005)
        inp = d["inputs"]                        # (N, 9005)
        N = sv.shape[0]
        atg_shap_runs.append(sv[:, :atg_size].reshape(N, n_ch, w_atg))
        stop_shap_runs.append(sv[:, atg_size:atg_size + stop_size].reshape(N, n_ch, w_stop))
        struct_shap_runs.append(sv[:, atg_size + stop_size:])
        atg_in_runs.append(inp[:, :atg_size].reshape(N, n_ch, w_atg))
        stop_in_runs.append(inp[:, atg_size:atg_size + stop_size].reshape(N, n_ch, w_stop))
        if labels is None:
            labels = d["labels"]
            channel_names = list(d["channel_names"])
            feature_names = list(d["feature_names"])
        print(f"  loaded run{r}: {path.name}  N={N}")

    atg_shap = np.mean(atg_shap_runs, axis=0)
    stop_shap = np.mean(stop_shap_runs, axis=0)
    struct_shap = np.mean(struct_shap_runs, axis=0)
    atg_inp = np.mean(atg_in_runs, axis=0)
    stop_inp = np.mean(stop_in_runs, axis=0)
    return dict(atg_shap=atg_shap, stop_shap=stop_shap, struct_shap=struct_shap,
                atg_inp=atg_inp, stop_inp=stop_inp,
                labels=labels, channel_names=channel_names, feature_names=feature_names,
                w_atg=w_atg, w_stop=w_stop, n_ch=n_ch)


def write_sample_shap_structural(meta, ids, struct_shap, labels, n_runs, out_path):
    df = pd.DataFrame(struct_shap, columns=meta["feature_names"])
    df.insert(0, "isoform_id", ids)
    df["label"] = labels.astype(int)
    df["n_runs"] = n_runs
    df.to_csv(out_path, sep="\t", index=False)
    print(f"  -> {out_path.name}  ({len(df)} rows)")


def write_class_profile(shap_arr, labels, channel_names, window, out_path):
    """Per-position, per-channel mean |SHAP| for NMD vs Control (no subgroup
    stratification). Schema: relative_position, channel, nmd_abs_shap, ctrl_abs_shap."""
    rel_pos = np.arange(window) - (window // 2)
    nmd_mask = labels > 0.5
    ctrl_mask = ~nmd_mask
    rows = []
    for ci, ch in enumerate(channel_names):
        nmd_mean = np.abs(shap_arr[nmd_mask, ci, :]).mean(axis=0)
        ctrl_mean = np.abs(shap_arr[ctrl_mask, ci, :]).mean(axis=0)
        for p, n_v, c_v in zip(rel_pos, nmd_mean, ctrl_mean):
            rows.append((int(p), ch, float(n_v), float(c_v)))
    out = pd.DataFrame(rows, columns=["relative_position", "channel",
                                       "nmd_abs_shap", "ctrl_abs_shap"])
    out.to_csv(out_path, sep="\t", index=False)
    print(f"  -> {out_path.name}  ({len(out)} rows)")


def write_subgroup_profile(shap_arr, subgroups, channel_names, n_ch, window,
                            window_name, out_path):
    """Per-subgroup, per-position total |SHAP| over the 4 nucleotide channels."""
    nuc_idx = [channel_names.index(c) for c in ["A", "C", "G", "T"]]
    nuc_abs = np.abs(shap_arr[:, nuc_idx, :])              # (N, 4, W)
    total = nuc_abs.sum(axis=1)                            # (N, W)
    rel_pos = np.arange(window) - (window // 2)            # -250..249

    rows = []
    for sg in pd.unique(subgroups):
        mask = subgroups == sg
        n = int(mask.sum())
        if n == 0:
            continue
        mean_per_pos = total[mask].mean(axis=0)
        for p, v in zip(rel_pos, mean_per_pos):
            rows.append((sg, int(p), float(v), n))
    out = pd.DataFrame(rows, columns=["subgroup", "relative_position",
                                       "total_nuc_abs_shap", "n_samples"])
    out.to_csv(out_path, sep="\t", index=False)
    print(f"  -> {out_path.name}  ({len(out)} rows, {window_name} branch)")


def write_subgroup_motif_logo(shap_arr, inp_arr, subgroups, channel_names,
                               n_ch, window, window_name, out_path,
                               half_width=15):
    """Per-subgroup, per-position, per-channel mean SHAP × input + freq."""
    nuc_idx = {c: channel_names.index(c) for c in ["A", "C", "G", "T"]}
    rel_pos = np.arange(window) - (window // 2)
    keep = (rel_pos >= -half_width) & (rel_pos <= half_width)

    contrib = shap_arr * inp_arr  # (N, n_ch, W)

    rows = []
    for sg in pd.unique(subgroups):
        mask = subgroups == sg
        n = int(mask.sum())
        if n == 0:
            continue
        for ch_name, ch_i in nuc_idx.items():
            mean_contrib = contrib[mask, ch_i, :].mean(axis=0)
            freq = inp_arr[mask, ch_i, :].mean(axis=0)
            for p in np.where(keep)[0]:
                rows.append((sg, int(rel_pos[p]), ch_name,
                             float(mean_contrib[p]), float(freq[p]), n))
    out = pd.DataFrame(rows, columns=["subgroup", "relative_position", "channel",
                                       "mean_contrib", "freq", "n_samples"])
    out.to_csv(out_path, sep="\t", index=False)
    print(f"  -> {out_path.name}  ({len(out)} rows, {window_name} branch)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None,
                                            help="Window-config tag. Default: the `selected:` block in --config. "
                                                 "Never a hardcoded literal -- see utils.selected_tag.")
    parser.add_argument("--config", default="config.yaml",
                           help="Where the selected window configuration is read from")
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--results-dir", default="results_4ct")
    args = parser.parse_args()

    # Resolve the tag from the ONE place that names the selected configuration.
    if args.tag is None:
        args.tag = selected_tag(load_config(args.config))
    results_dir = Path(args.results_dir)
    tag = args.tag

    # ── Subgroup classifications ──
    print("Loading test predictions and reference/TD2 features ...")
    preds = pd.read_csv(results_dir / f"predictions_{tag}.tsv", sep="\t",
                        dtype={"isoform_id": str})
    ref = pd.read_csv(results_dir / "ref_cds_features.tsv", sep="\t",
                      dtype={"isoform_id": str})
    td2 = pd.read_csv(results_dir / "td2_features.tsv", sep="\t",
                      dtype={"isoform_id": str})

    sg = (preds[["isoform_id", "label"]]
          .merge(ref[["isoform_id", "category"]], on="isoform_id", how="left")
          .merge(td2[["isoform_id", "td2_downstream_ejc"]], on="isoform_id", how="left"))
    sg["subgroup"] = sg.apply(assign_subgroup, axis=1)
    sg = sg.set_index("isoform_id").reindex(preds["isoform_id"]).reset_index()
    subgroups = sg["subgroup"].values
    print(f"  subgroups: {dict(pd.Series(subgroups).value_counts())}")

    # ── Load DeepSHAP NPZs averaged across runs ──
    print(f"\nLoading {args.n_runs} joint DeepSHAP NPZs for tag={tag} ...")
    meta = load_runs(results_dir, tag, args.n_runs)

    # NPZ explain_indices were verified to be 0..N-1 sorted, so NPZ row i ↔ preds row i.
    isoform_ids = preds["isoform_id"].values
    assert len(isoform_ids) == meta["atg_shap"].shape[0], "test set size mismatch"

    # ── Outputs ──
    print("\nWriting outputs ...")
    write_sample_shap_structural(
        meta, isoform_ids, meta["struct_shap"], meta["labels"], args.n_runs,
        results_dir / f"sample_shap_structural_{tag}.tsv")

    write_class_profile(
        meta["atg_shap"], meta["labels"], meta["channel_names"], meta["w_atg"],
        results_dir / f"shap_profile_atg_joint_{tag}.tsv")

    write_class_profile(
        meta["stop_shap"], meta["labels"], meta["channel_names"], meta["w_stop"],
        results_dir / f"shap_profile_stop_joint_{tag}.tsv")

    write_subgroup_profile(
        meta["atg_shap"], subgroups, meta["channel_names"], meta["n_ch"],
        meta["w_atg"], "ATG",
        results_dir / f"shap_profile_atg_subgroup_joint_{tag}.tsv")

    write_subgroup_profile(
        meta["stop_shap"], subgroups, meta["channel_names"], meta["n_ch"],
        meta["w_stop"], "stop",
        results_dir / f"shap_profile_stop_subgroup_joint_{tag}.tsv")

    write_subgroup_motif_logo(
        meta["atg_shap"], meta["atg_inp"], subgroups, meta["channel_names"],
        meta["n_ch"], meta["w_atg"], "ATG",
        results_dir / f"motif_logo_atg_subgroup_joint_{tag}.tsv")

    write_subgroup_motif_logo(
        meta["stop_shap"], meta["stop_inp"], subgroups, meta["channel_names"],
        meta["n_ch"], meta["w_stop"], "stop",
        results_dir / f"motif_logo_stop_subgroup_joint_{tag}.tsv")

    print("\nDone.")


if __name__ == "__main__":
    main()

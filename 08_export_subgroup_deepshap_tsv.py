#!/usr/bin/env python3
"""
08_export_subgroup_deepshap_tsv.py — Export subgroup-stratified DeepSHAP statistics.

Processes deepshap_{atg,stop}_atg{A}_stop{S}_run{1-5}.npz and exports per-subgroup
SHAP statistics as TSVs for R consumption in the v5 report.

Subgroup assignment mirrors the R report logic (orf_model_report_v5.Rmd lines 1565-1581),
including the TD2 reclassification of ref_atg_lost with td2_downstream_ejc > 0 → PTC+.

Usage:
    python 08_export_subgroup_deepshap_tsv.py --atg 500 --stop 500 --n-runs 5
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Subgroup assignment (must mirror R report exactly)
# ---------------------------------------------------------------------------
PTC_PLUS_CATS = {"effectively_ptc"}
ATG_RETAINED_CATS = {"no_downstream_ejc", "truncated_no_ejc"}
ATG_LOST_CATS = {"ref_atg_lost", "no_ref_isoform", "not_atg_in_target", "no_stop_in_target"}


def assign_subgroups(isoform_ids, labels, ref_features, td2_features):
    """
    Assign NMD subgroups matching R report logic.

    Returns: pd.Series of subgroup labels indexed by isoform_id.
    """
    df = pd.DataFrame({"isoform_id": isoform_ids, "label": labels})
    df = df.merge(ref_features[["isoform_id", "category"]], on="isoform_id", how="left")
    df = df.merge(td2_features[["isoform_id", "td2_downstream_ejc"]], on="isoform_id", how="left")

    def _assign(row):
        if row["label"] == 0:
            return "Control"
        cat = row["category"]
        if cat in PTC_PLUS_CATS:
            return "NMD PTC+"
        # TD2 reclassification: ref_atg_lost with td2_downstream_ejc > 0 → PTC+
        if cat in ATG_LOST_CATS:
            if pd.notna(row["td2_downstream_ejc"]) and row["td2_downstream_ejc"] > 0:
                return "NMD PTC+"
            return "NMD PTC-, ref ATG lost"
        if cat in ATG_RETAINED_CATS:
            return "NMD PTC-, ref ATG retained"
        return "NMD other"

    df["subgroup"] = df.apply(_assign, axis=1)
    result = df.set_index("isoform_id")["subgroup"]
    assert len(result) == len(isoform_ids), \
        f"Subgroup map length ({len(result)}) != input length ({len(isoform_ids)}). " \
        "Possible duplicate isoform_ids in ref/td2 features."
    return result


# ---------------------------------------------------------------------------
# NPZ loading
# ---------------------------------------------------------------------------
def load_npz(path):
    """Load NPZ, squeeze trailing singleton, return (shap, inputs, labels, explain_indices)."""
    data = np.load(path, allow_pickle=True)
    shap_vals = data["shap_values"]
    if shap_vals.ndim == 4 and shap_vals.shape[-1] == 1:
        shap_vals = shap_vals.squeeze(-1)
    return shap_vals, data["inputs"], data["labels"], data["explain_indices"]


# ---------------------------------------------------------------------------
# Kozak analysis
# ---------------------------------------------------------------------------
def kozak_to_array_pos(kozak_pos, atg_pos):
    """Convert standard Kozak position to array position.

    Standard Kozak: A of ATG = +1, no position 0.
    In the array: A of ATG is at atg_pos - 1 (one before the T at center).
    """
    # A of ATG is at array position (atg_pos - 1), which is Kozak +1
    a_of_atg = atg_pos - 1
    if kozak_pos >= 1:
        return a_of_atg + (kozak_pos - 1)  # +1→a_of_atg, +2→a_of_atg+1, etc.
    else:
        return a_of_atg + kozak_pos  # -1→a_of_atg-1, -3→a_of_atg-3, etc.


def compute_kozak_subgroup(shap, inputs, subgroups, atg_pos):
    """Compute per-nucleotide SHAP at Kozak positions by subgroup."""
    nucs = ["A", "C", "G", "T"]
    # Kozak positions: -6 to -1, +1 to +7 (skipping 0)
    kozak_positions = list(range(-6, 0)) + list(range(1, 8))
    rows = []
    for sg in ["NMD PTC+", "NMD PTC-, ref ATG retained", "NMD PTC-, ref ATG lost", "Control"]:
        mask = subgroups == sg
        n = mask.sum()
        if n == 0:
            continue
        for kpos in kozak_positions:
            arr_pos = kozak_to_array_pos(kpos, atg_pos)
            if arr_pos < 0 or arr_pos >= shap.shape[2]:
                continue
            for ch_idx, nuc in enumerate(nucs):
                gxi = shap[mask, ch_idx, arr_pos] * inputs[mask, ch_idx, arr_pos]
                freq = inputs[mask, ch_idx, arr_pos].mean()
                rows.append({
                    "subgroup": sg,
                    "kozak_position": kpos,
                    "channel": nuc,
                    "mean_signed_shap": gxi.mean(),
                    "mean_abs_shap": np.abs(gxi).mean(),
                    "freq": freq,
                    "n_samples": int(n),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5'UTR channel SHAP
# ---------------------------------------------------------------------------
CHANNELS = ["A", "C", "G", "T", "junction", "rolling_gc", "frame_0", "frame_1", "frame_2"]

def compute_utr5_channel_shap(shap, inputs, subgroups, atg_pos):
    """Mean channel SHAP in 5'UTR region (upstream of Kozak zone)."""
    # Exclude Kozak ±6bp zone: use positions 0 to (atg_pos - 7)
    utr5_end = max(atg_pos - 6, 0)
    if utr5_end == 0:
        return pd.DataFrame()

    rows = []
    for sg in ["NMD PTC+", "NMD PTC-, ref ATG retained", "NMD PTC-, ref ATG lost", "Control"]:
        mask = subgroups == sg
        n = mask.sum()
        if n == 0:
            continue
        for ch_idx, ch_name in enumerate(CHANNELS):
            gxi = shap[mask, ch_idx, :utr5_end] * inputs[mask, ch_idx, :utr5_end]
            per_sample = gxi.mean(axis=1)  # mean across positions per sample
            rows.append({
                "subgroup": sg,
                "channel": ch_name,
                "mean_signed_shap": per_sample.mean(),
                "mean_abs_shap": np.abs(gxi).mean(),
                "n_samples": int(n),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stop codon identity
# ---------------------------------------------------------------------------
def compute_stop_codon_shap(shap, inputs, subgroups, stop_pos):
    """Stop codon identity (TGA/TAA/TAG) frequency and SHAP by subgroup."""
    # Stop codon layout: stop_pos-1 = T (constant), stop_pos = A or G, stop_pos+1 = A or G
    # (window is centered on the middle nucleotide of the stop codon)
    pos1 = stop_pos
    pos2 = stop_pos + 1
    if pos2 >= shap.shape[2]:
        return pd.DataFrame()

    # Determine stop codon type per sample
    p1_A = inputs[:, 0, pos1] > 0.5
    p1_G = inputs[:, 2, pos1] > 0.5
    p2_A = inputs[:, 0, pos2] > 0.5
    p2_G = inputs[:, 2, pos2] > 0.5

    stop_type = np.where(
        p1_A & p2_A, "TAA",
        np.where(p1_A & p2_G, "TAG",
        np.where(p1_G & p2_A, "TGA", "other")))

    rows = []
    for sg in ["NMD PTC+", "NMD PTC-, ref ATG retained", "NMD PTC-, ref ATG lost", "Control"]:
        mask = subgroups == sg
        n = mask.sum()
        if n == 0:
            continue
        for sc in ["TAA", "TAG", "TGA"]:
            sc_mask = mask & (stop_type == sc)
            n_sc = sc_mask.sum()
            if n_sc == 0:
                rows.append({
                    "subgroup": sg, "stop_codon": sc,
                    "n": 0, "pct": 0.0, "mean_signed_shap": np.nan,
                    "n_total": int(n),
                })
                continue
            # Total signed SHAP at variable positions (pos1 + pos2)
            total_shap = sum(
                (shap[sc_mask, ch, pos1] * inputs[sc_mask, ch, pos1]).mean() +
                (shap[sc_mask, ch, pos2] * inputs[sc_mask, ch, pos2]).mean()
                for ch in range(4)
            )
            rows.append({
                "subgroup": sg, "stop_codon": sc,
                "n": int(n_sc), "pct": n_sc / n,
                "mean_signed_shap": total_shap,
                "n_total": int(n),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Downstream junction SHAP
# ---------------------------------------------------------------------------
def compute_junction_downstream_shap(shap, inputs, subgroups, stop_pos):
    """Junction channel SHAP downstream of stop by subgroup."""
    downstream_start = stop_pos + 3  # after stop codon
    if downstream_start >= shap.shape[2]:
        return pd.DataFrame()

    junc_ch = CHANNELS.index("junction")
    rows = []
    for sg in ["NMD PTC+", "NMD PTC-, ref ATG retained", "NMD PTC-, ref ATG lost", "Control"]:
        mask = subgroups == sg
        n = mask.sum()
        if n == 0:
            continue
        gxi = shap[mask, junc_ch, downstream_start:] * inputs[mask, junc_ch, downstream_start:]
        per_sample = gxi.mean(axis=1)
        junc_freq = inputs[mask, junc_ch, downstream_start:].mean()
        junc_any = (inputs[mask, junc_ch, downstream_start:].sum(axis=1) > 0).mean()
        junc_count = inputs[mask, junc_ch, downstream_start:].sum(axis=1).mean()
        rows.append({
            "subgroup": sg,
            "mean_junction_shap": per_sample.mean(),
            "frac_with_junction": float(junc_any),
            "mean_junc_count": float(junc_count),
            "n_samples": int(n),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Frame periodicity (negative finding)
# ---------------------------------------------------------------------------
def compute_frame_periodicity(shap, inputs, subgroups, stop_pos):
    """Frame position SHAP and autocorrelation for codon periodicity test."""
    # Use ORF body: positions 100 to stop_pos (avoid edge effects)
    body_start = min(100, stop_pos - 50)
    body_end = stop_pos

    rows = []
    for sg in ["NMD PTC+", "NMD PTC-, ref ATG retained", "NMD PTC-, ref ATG lost", "Control"]:
        mask = subgroups == sg
        n = mask.sum()
        if n == 0:
            continue

        # Per-frame SHAP
        for frame in [0, 1, 2]:
            positions = np.array([p for p in range(body_start, body_end) if (stop_pos - p) % 3 == frame])
            if len(positions) == 0:
                continue
            nuc_shap = sum(
                np.abs(shap[mask][:, ch, positions] * inputs[mask][:, ch, positions]).mean()
                for ch in range(4)
            )
            rows.append({
                "subgroup": sg, "frame": frame,
                "mean_abs_shap": nuc_shap, "n_positions": len(positions),
                "n_samples": int(n),
            })

        # Autocorrelation of positional SHAP
        nuc_shap_pos = np.zeros(body_end - body_start)
        for ch in range(4):
            nuc_shap_pos += np.abs(
                shap[mask, ch, body_start:body_end] * inputs[mask, ch, body_start:body_end]
            ).mean(axis=0)
        nuc_dt = nuc_shap_pos - nuc_shap_pos.mean()
        if len(nuc_dt) > 6:
            lag1 = np.corrcoef(nuc_dt[:-1], nuc_dt[1:])[0, 1]
            lag3 = np.corrcoef(nuc_dt[:-3], nuc_dt[3:])[0, 1]
        else:
            lag1 = lag3 = np.nan

        # Update frame rows with autocorrelation (add to frame=0 row)
        for r in rows:
            if r["subgroup"] == sg and r["frame"] == 0:
                r["lag1_autocorr"] = lag1
                r["lag3_autocorr"] = lag3

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rolling GC channel by region
# ---------------------------------------------------------------------------
def compute_gc_channel_by_region(shap_atg, inputs_atg, shap_stop, inputs_stop,
                                  subgroups_atg, subgroups_stop, atg_pos, stop_pos):
    """Rolling GC channel importance by region and subgroup."""
    gc_ch = CHANNELS.index("rolling_gc")
    regions = [
        ("5UTR", shap_atg, inputs_atg, subgroups_atg, slice(0, atg_pos - 6)),
        ("CDS_near_ATG", shap_atg, inputs_atg, subgroups_atg, slice(atg_pos + 4, None)),
        ("ORF_body", shap_stop, inputs_stop, subgroups_stop, slice(0, stop_pos)),
        ("3UTR", shap_stop, inputs_stop, subgroups_stop, slice(stop_pos + 3, None)),
    ]
    rows = []
    for region_name, s, inp, sgs, sl in regions:
        for sg in ["NMD PTC+", "NMD PTC-, ref ATG retained", "NMD PTC-, ref ATG lost", "Control"]:
            mask = sgs == sg
            n = mask.sum()
            if n == 0:
                continue
            gxi_gc = s[mask, gc_ch, sl] * inp[mask, gc_ch, sl]
            # Total ACGT for ratio
            nuc_total = sum(np.abs(s[mask, ch, sl] * inp[mask, ch, sl]).mean() for ch in range(4))
            gc_abs = np.abs(gxi_gc).mean()
            pct = gc_abs / (nuc_total + gc_abs) * 100 if (nuc_total + gc_abs) > 0 else 0
            rows.append({
                "subgroup": sg, "region": region_name,
                "mean_abs_shap": gc_abs,
                "pct_of_total": pct,
                "mean_signed_shap": gxi_gc.mean(),
                "n_samples": int(n),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Export subgroup-specific DeepSHAP TSVs")
    parser.add_argument("--atg", type=int, default=500)
    parser.add_argument("--stop", type=int, default=500)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    tag = f"atg{args.atg}_stop{args.stop}"
    results_dir = Path(args.results_dir)
    h5_path = results_dir / "nmd_orf_data.h5"

    # ── Load reference data ──
    print("Loading reference data ...")
    ref_features = pd.read_csv(results_dir / "ref_cds_features.tsv", sep="\t")
    td2_features = pd.read_csv(results_dir / "td2_features.tsv", sep="\t")
    preds = pd.read_csv(results_dir / f"predictions_{tag}.tsv", sep="\t")

    # ── Load test set isoform IDs from HDF5 ──
    print("Loading test set isoform IDs from HDF5 ...")
    with h5py.File(h5_path, "r") as f:
        splits = np.array([s.decode() if isinstance(s, bytes) else s for s in f["split"][:]])
        all_ids = np.array([s.decode() if isinstance(s, bytes) else s for s in f["isoform_id"][:]])
        test_mask = splits == "test"
        test_indices = np.where(test_mask)[0]
        test_ids = all_ids[test_indices]

    # Verify alignment with predictions
    assert np.all(test_ids == preds["isoform_id"].values), \
        "HDF5 test set order does not match predictions file!"

    # ── Assign subgroups ──
    subgroup_map = assign_subgroups(
        test_ids, preds["label"].values, ref_features, td2_features
    )
    sg_counts = subgroup_map.value_counts()
    print("\nSubgroup distribution (full test set):")
    for sg, n in sg_counts.items():
        print(f"  {sg}: {n}")

    n_other = (subgroup_map == "NMD other").sum()
    if n_other > 0:
        print(f"\n  WARNING: {n_other} 'NMD other' isoforms excluded from subgroup analyses")

    # ── Process each run ──
    all_kozak = []
    all_utr5 = []
    all_stop_codon = []
    all_junction = []
    all_frame = []
    all_gc = []

    for run in range(1, args.n_runs + 1):
        print(f"\n{'='*50}")
        print(f"Processing run {run}/{args.n_runs}")
        print(f"{'='*50}")

        atg_path = results_dir / f"deepshap_atg_{tag}_run{run}.npz"
        stop_path = results_dir / f"deepshap_stop_{tag}_run{run}.npz"

        if not atg_path.exists() or not stop_path.exists():
            print(f"  SKIP: {atg_path} or {stop_path} not found")
            continue

        shap_atg, inp_atg, labels_atg, idx_atg = load_npz(atg_path)
        shap_stop, inp_stop, labels_stop, idx_stop = load_npz(stop_path)

        # Map explain_indices to subgroups
        # explain_indices index into the test dataset (same order as predictions)
        sg_atg = subgroup_map.iloc[idx_atg].values
        sg_stop = subgroup_map.iloc[idx_stop].values

        # Verify label consistency
        assert np.all(labels_atg == preds["label"].values[idx_atg]), \
            f"ATG labels mismatch in run {run}!"
        assert np.all(labels_stop == preds["label"].values[idx_stop]), \
            f"STOP labels mismatch in run {run}!"

        n_nmd_atg = (labels_atg == 1).sum()
        n_nmd_stop = (labels_stop == 1).sum()
        print(f"  ATG: {len(idx_atg)} samples ({n_nmd_atg} NMD)")
        print(f"  STOP: {len(idx_stop)} samples ({n_nmd_stop} NMD)")

        # Subgroup counts in this run's DeepSHAP subset
        for sg in ["NMD PTC+", "NMD PTC-, ref ATG retained", "NMD PTC-, ref ATG lost", "Control"]:
            n_sg = ((sg_atg == sg) & (labels_atg == (0 if sg == "Control" else 1))).sum()
            print(f"    {sg}: {n_sg}")

        atg_pos = shap_atg.shape[2] // 2  # ATG at window center
        stop_pos = shap_stop.shape[2] // 2  # stop codon at window center

        # Kozak
        kozak_df = compute_kozak_subgroup(shap_atg, inp_atg, sg_atg, atg_pos)
        kozak_df["run"] = run
        all_kozak.append(kozak_df)

        # 5'UTR channel SHAP
        utr5_df = compute_utr5_channel_shap(shap_atg, inp_atg, sg_atg, atg_pos)
        utr5_df["run"] = run
        all_utr5.append(utr5_df)

        # Stop codon identity
        stop_df = compute_stop_codon_shap(shap_stop, inp_stop, sg_stop, stop_pos)
        stop_df["run"] = run
        all_stop_codon.append(stop_df)

        # Junction downstream
        junc_df = compute_junction_downstream_shap(shap_stop, inp_stop, sg_stop, stop_pos)
        junc_df["run"] = run
        all_junction.append(junc_df)

        # Frame periodicity
        frame_df = compute_frame_periodicity(shap_stop, inp_stop, sg_stop, stop_pos)
        frame_df["run"] = run
        all_frame.append(frame_df)

        # GC channel by region
        gc_df = compute_gc_channel_by_region(
            shap_atg, inp_atg, shap_stop, inp_stop,
            sg_atg, sg_stop, atg_pos, stop_pos
        )
        gc_df["run"] = run
        all_gc.append(gc_df)

    # ── Aggregate across runs ──
    print(f"\n{'='*50}")
    print("Aggregating across runs and computing stability ...")
    print(f"{'='*50}")

    def aggregate_and_save(run_dfs, key_cols, value_cols, out_name):
        """Aggregate per-run data: compute mean, SE, CI, CV, sign consistency."""
        combined = pd.concat(run_dfs, ignore_index=True)

        # Per-run means
        run_means = combined.groupby(key_cols + ["run"])[value_cols].mean().reset_index()

        # Cross-run statistics
        agg = run_means.groupby(key_cols)[value_cols].agg(["mean", "std", "count"]).reset_index()

        # Flatten multi-level columns
        flat_cols = []
        for col in agg.columns:
            if isinstance(col, tuple):
                if col[1]:
                    flat_cols.append(f"{col[0]}_{col[1]}")
                else:
                    flat_cols.append(col[0])
            else:
                flat_cols.append(col)
        agg.columns = flat_cols

        # Compute CV and sign consistency for each value column
        for vc in value_cols:
            mean_col = f"{vc}_mean"
            std_col = f"{vc}_std"
            if mean_col in agg.columns and std_col in agg.columns:
                agg[f"{vc}_cv"] = agg[std_col] / agg[mean_col].abs()
                agg[f"{vc}_cv"] = agg[f"{vc}_cv"].replace([np.inf, -np.inf], np.nan)

                # Sign consistency
                sign_data = run_means.groupby(key_cols)[vc].apply(
                    lambda x: "YES" if all(x > 0) or all(x < 0) else "NO"
                ).reset_index(name=f"{vc}_sign_consistent")
                agg = agg.merge(sign_data, on=key_cols, how="left")

                # SE and CI (across runs)
                count_col = f"{vc}_count"
                n_runs_per_row = agg[count_col] if count_col in agg.columns else len(run_dfs)
                agg[f"{vc}_se"] = agg[std_col] / np.sqrt(n_runs_per_row)
                agg[f"{vc}_ci_lo"] = agg[mean_col] - 1.96 * agg[f"{vc}_se"]
                agg[f"{vc}_ci_hi"] = agg[mean_col] + 1.96 * agg[f"{vc}_se"]

        # Include per-subgroup n_samples (mean across runs in case of variation)
        if "n_samples" in combined.columns:
            n_info = combined.groupby(key_cols)["n_samples"].mean().reset_index()
            n_info["n_samples"] = n_info["n_samples"].round().astype(int)
            agg = agg.merge(n_info, on=key_cols, how="left")

        # Save
        path = results_dir / f"{out_name}_{tag}.tsv"
        agg.to_csv(path, sep="\t", index=False)
        print(f"  -> {path} ({len(agg)} rows)")
        return agg

    # Kozak
    if all_kozak:
        aggregate_and_save(
            all_kozak, ["subgroup", "kozak_position", "channel"],
            ["mean_signed_shap", "mean_abs_shap", "freq"],
            "subgroup_kozak_shap"
        )

    # 5'UTR
    if all_utr5:
        aggregate_and_save(
            all_utr5, ["subgroup", "channel"],
            ["mean_signed_shap", "mean_abs_shap"],
            "subgroup_utr5_channel_shap"
        )

    # Stop codon
    if all_stop_codon:
        aggregate_and_save(
            all_stop_codon, ["subgroup", "stop_codon"],
            ["mean_signed_shap", "pct"],
            "subgroup_stop_codon_shap"
        )

    # Junction downstream
    if all_junction:
        aggregate_and_save(
            all_junction, ["subgroup"],
            ["mean_junction_shap", "frac_with_junction", "mean_junc_count"],
            "subgroup_junction_downstream_shap"
        )

    # Frame periodicity
    if all_frame:
        aggregate_and_save(
            all_frame, ["subgroup", "frame"],
            ["mean_abs_shap"],
            "subgroup_frame_periodicity"
        )

    # GC channel
    if all_gc:
        aggregate_and_save(
            all_gc, ["subgroup", "region"],
            ["mean_abs_shap", "pct_of_total", "mean_signed_shap"],
            "subgroup_gc_channel"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

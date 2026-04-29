#!/usr/bin/env python3
"""
07_motif_analysis.py — Per-nucleotide SHAP motif analysis for ATG and STOP branches.

Approach A: Multi-resolution logos at biologically meaningful landmarks.
  - Full-resolution SHAP×input logos at ±window around:
    - Stop codon triplet
    - Exon-exon junctions in the 3'UTR
    - First downstream ATG (reinitiation)
  - Also: ATG branch full logo (already small, 20-100bp)

Approach B: Motif discovery from high-SHAP positions.
  - Identify positions with top |SHAP| values
  - Extract surrounding k-mers
  - Compute k-mer frequency enrichment in NMD vs control
  - Cluster high-SHAP k-mers to find recurring motifs

Outputs TSVs that the RMarkdown report can consume.
"""

import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


def load_npz(path):
    """Load NPZ and squeeze trailing singleton dim from shap_values."""
    data = np.load(path, allow_pickle=True)
    shap = data["shap_values"]
    if shap.ndim == 4 and shap.shape[-1] == 1:
        shap = shap.squeeze(-1)
    return shap, data["inputs"], data["labels"], list(data["channel_names"])


def onehot_to_seq(inputs_2d):
    """Convert (4, W) one-hot nucleotide channels to a sequence string."""
    nucs = np.array(["A", "C", "G", "T"])
    # inputs_2d shape: (4, W) — first 4 channels are nucleotides
    idx = np.argmax(inputs_2d, axis=0)
    # Handle positions where all channels are 0 (padding)
    max_vals = np.max(inputs_2d, axis=0)
    seq = []
    for i in range(inputs_2d.shape[1]):
        if max_vals[i] < 0.5:
            seq.append("N")
        else:
            seq.append(nucs[idx[i]])
    return "".join(seq)


# =============================================================================
# Approach A: Multi-resolution logos at landmarks
# =============================================================================

def find_stop_position(inputs, channels):
    """Stop codon is at window center in v5."""
    return inputs.shape[2] // 2


def find_junctions_per_sample(inputs, channels, start_pos):
    """Find junction positions downstream of start_pos for each sample."""
    junc_idx = channels.index("junction")
    junc_data = inputs[:, junc_idx, :]  # (N, W)
    results = []
    for i in range(len(inputs)):
        positions = np.where(junc_data[i, start_pos:] > 0.5)[0] + start_pos
        results.append(positions.tolist())
    return results


def find_first_downstream_atg(inputs, channels, start_pos):
    """Find first ATG trinucleotide downstream of start_pos from nucleotide channels."""
    # v5: no ATG channel — detect ATG from A,C,G,T one-hot channels
    a_idx = channels.index("A")
    t_idx = channels.index("T")
    g_idx = channels.index("G")
    results = []
    W = inputs.shape[2]
    for i in range(len(inputs)):
        found = None
        for p in range(start_pos, W - 2):
            if (inputs[i, a_idx, p] > 0.5 and
                inputs[i, t_idx, p+1] > 0.5 and
                inputs[i, g_idx, p+2] > 0.5):
                found = p
                break
        results.append(found)
    return results


def extract_logo_at_landmark(shap, inputs, labels, center_positions,
                              half_window=15, label="landmark"):
    """
    Extract per-nucleotide SHAP×input logo data centered on variable positions.

    center_positions: list of int or None per sample. None = skip.
    Returns DataFrame with columns: relative_position, channel, nmd/ctrl values.
    """
    nucs = ["A", "C", "G", "T"]
    nmd = labels == 1
    ctrl = labels == 0
    W = shap.shape[2]

    # Collect contributions aligned to the landmark
    window_len = 2 * half_window + 1
    nmd_contribs = np.zeros((nmd.sum(), 4, window_len))
    ctrl_contribs = np.zeros((ctrl.sum(), 4, window_len))
    nmd_freqs = np.zeros((nmd.sum(), 4, window_len))
    ctrl_freqs = np.zeros((ctrl.sum(), 4, window_len))

    nmd_count = 0
    ctrl_count = 0
    nmd_used = 0
    ctrl_used = 0

    for i in range(len(labels)):
        center = center_positions[i]
        if center is None:
            if nmd[i]:
                nmd_count += 1
            else:
                ctrl_count += 1
            continue

        lo = center - half_window
        hi = center + half_window + 1

        # Skip if window falls outside the sequence
        if lo < 0 or hi > W:
            if nmd[i]:
                nmd_count += 1
            else:
                ctrl_count += 1
            continue

        contrib = shap[i, :4, lo:hi] * inputs[i, :4, lo:hi]
        freq = inputs[i, :4, lo:hi]

        if nmd[i]:
            nmd_contribs[nmd_used] = contrib
            nmd_freqs[nmd_used] = freq
            nmd_used += 1
        else:
            ctrl_contribs[ctrl_used] = contrib
            ctrl_freqs[ctrl_used] = freq
            ctrl_used += 1

    # Trim to actual count
    nmd_contribs = nmd_contribs[:nmd_used]
    ctrl_contribs = ctrl_contribs[:ctrl_used]
    nmd_freqs = nmd_freqs[:nmd_used]
    ctrl_freqs = ctrl_freqs[:ctrl_used]

    rows = []
    for pos_idx in range(window_len):
        rel_pos = pos_idx - half_window
        for ch_idx, ch in enumerate(nucs):
            row = {
                "landmark": label,
                "relative_position": rel_pos,
                "channel": ch,
                "nmd_mean_contrib": float(nmd_contribs[:, ch_idx, pos_idx].mean()) if nmd_used > 0 else 0,
                "ctrl_mean_contrib": float(ctrl_contribs[:, ch_idx, pos_idx].mean()) if ctrl_used > 0 else 0,
                "nmd_mean_abs_shap": float(np.abs(nmd_contribs[:, ch_idx, pos_idx]).mean()) if nmd_used > 0 else 0,
                "nmd_freq": float(nmd_freqs[:, ch_idx, pos_idx].mean()) if nmd_used > 0 else 0,
                "ctrl_freq": float(ctrl_freqs[:, ch_idx, pos_idx].mean()) if ctrl_used > 0 else 0,
                "n_nmd": nmd_used,
                "n_ctrl": ctrl_used,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def extract_logo_fixed_position(shap, inputs, labels, center_pos, half_window=15,
                                 label="landmark"):
    """Extract logo at a fixed position (same for all samples)."""
    positions = [center_pos] * len(labels)
    return extract_logo_at_landmark(shap, inputs, labels, positions,
                                     half_window=half_window, label=label)


def approach_a_logos(shap, inputs, labels, channels, stop_pos, out_dir, tag):
    """Generate multi-resolution logo TSVs at biologically meaningful landmarks."""
    nmd = labels == 1
    print(f"\n=== Approach A: Multi-resolution logos ===")

    all_logos = []

    # 1. Stop codon logo (±15bp)
    print(f"  Stop codon at position {stop_pos}")
    logo_stop = extract_logo_fixed_position(shap, inputs, labels, stop_pos,
                                             half_window=15, label="stop_codon")
    all_logos.append(logo_stop)

    # 2. First exon-exon junction in 3'UTR (per-sample variable position)
    junctions = find_junctions_per_sample(inputs, channels, stop_pos + 1)
    first_junc = [js[0] if len(js) > 0 else None for js in junctions]
    n_with_junc = sum(1 for j in first_junc if j is not None)
    print(f"  First 3'UTR junction: {n_with_junc}/{len(labels)} samples have one")
    if n_with_junc > 50:
        logo_junc = extract_logo_at_landmark(shap, inputs, labels, first_junc,
                                              half_window=15, label="first_3utr_junction")
        all_logos.append(logo_junc)

    # 3. Second junction (if available — captures more distal EJC signal)
    second_junc = [js[1] if len(js) > 1 else None for js in junctions]
    n_with_junc2 = sum(1 for j in second_junc if j is not None)
    print(f"  Second 3'UTR junction: {n_with_junc2}/{len(labels)} samples have one")
    if n_with_junc2 > 50:
        logo_junc2 = extract_logo_at_landmark(shap, inputs, labels, second_junc,
                                               half_window=15, label="second_3utr_junction")
        all_logos.append(logo_junc2)

    # 4. 50bp into 3'UTR (fixed, captures early UTR composition)
    if stop_pos + 50 < shap.shape[2]:
        logo_utr50 = extract_logo_fixed_position(shap, inputs, labels, stop_pos + 50,
                                                   half_window=15, label="3utr_+50bp")
        all_logos.append(logo_utr50)

    # 5. 150bp into 3'UTR
    if stop_pos + 150 < shap.shape[2]:
        logo_utr150 = extract_logo_fixed_position(shap, inputs, labels, stop_pos + 150,
                                                    half_window=15, label="3utr_+150bp")
        all_logos.append(logo_utr150)

    combined = pd.concat(all_logos, ignore_index=True)
    path = out_dir / f"motif_logos_stop_{tag}.tsv"
    combined.to_csv(path, sep="\t", index=False)
    print(f"  -> {path} ({len(combined)} rows, {combined['landmark'].nunique()} landmarks)")

    return combined


def approach_a_atg_logo(shap, inputs, labels, channels, out_dir, tag):
    """Full-resolution ATG logo (entire window — already small)."""
    print(f"\n=== ATG full-resolution logo ===")
    nucs = ["A", "C", "G", "T"]
    nmd = labels == 1
    ctrl = labels == 0
    W = shap.shape[2]

    # ATG is at window center in v5
    atg_pos = W // 2
    print(f"  ATG at position {atg_pos} (window center), window size = {W}")

    rows = []
    for pos in range(W):
        rel = pos - atg_pos
        for ch_idx, ch in enumerate(nucs):
            nmd_contrib = float(np.mean(shap[nmd, ch_idx, pos] * inputs[nmd, ch_idx, pos]))
            ctrl_contrib = float(np.mean(shap[ctrl, ch_idx, pos] * inputs[ctrl, ch_idx, pos]))
            nmd_abs = float(np.mean(np.abs(shap[nmd, ch_idx, pos])))
            ctrl_abs = float(np.mean(np.abs(shap[ctrl, ch_idx, pos])))
            nmd_freq = float(np.mean(inputs[nmd, ch_idx, pos]))
            ctrl_freq = float(np.mean(inputs[ctrl, ch_idx, pos]))
            rows.append({
                "relative_position": rel,
                "channel": ch,
                "nmd_mean_contrib": nmd_contrib,
                "ctrl_mean_contrib": ctrl_contrib,
                "nmd_mean_abs_shap": nmd_abs,
                "ctrl_mean_abs_shap": ctrl_abs,
                "nmd_freq": nmd_freq,
                "ctrl_freq": ctrl_freq,
                "n_nmd": int(nmd.sum()),
                "n_ctrl": int(ctrl.sum()),
            })

    df = pd.DataFrame(rows)
    path = out_dir / f"motif_logo_atg_{tag}.tsv"
    df.to_csv(path, sep="\t", index=False)
    print(f"  -> {path} ({len(df)} rows)")
    return df


# =============================================================================
# Approach B: Motif discovery from high-SHAP positions
# =============================================================================

def approach_b_kmer_enrichment(shap, inputs, labels, channels, stop_pos,
                                out_dir, tag, k=8, n_top=50):
    """
    Identify high-|SHAP| positions, extract k-mers, compare NMD vs ctrl.

    For each sample:
      1. Compute per-position total |SHAP| across 4 nucleotide channels
      2. Identify top-N positions by |SHAP|
      3. Extract k-mer centered on each top position
      4. Compare k-mer frequencies between NMD and control samples
    """
    print(f"\n=== Approach B: k-mer enrichment (k={k}, top {n_top} positions) ===")
    nucs = ["A", "C", "G", "T"]
    nmd = labels == 1
    ctrl = labels == 0
    W = shap.shape[2]
    half_k = k // 2

    # Per-position total |SHAP| across nucleotide channels
    nuc_shap_abs = np.abs(shap[:, :4, :]).sum(axis=1)  # (N, W)

    # Restrict to 3'UTR region (downstream of stop)
    utr_start = stop_pos + 1
    if utr_start >= W - k:
        print("  Stop codon too close to end of window for 3'UTR k-mer analysis")
        return None

    nuc_shap_utr = nuc_shap_abs[:, utr_start:]
    utr_len = nuc_shap_utr.shape[1]

    # Also extract the ORF-tail region for comparison
    orf_start = max(0, stop_pos - 100)

    # For each sample, get top positions and extract k-mers
    nmd_kmers = Counter()
    ctrl_kmers = Counter()
    nmd_high_shap_kmers = Counter()
    ctrl_high_shap_kmers = Counter()

    # Also collect all k-mers (background) for enrichment
    nmd_all_kmers = Counter()
    ctrl_all_kmers = Counter()

    for i in range(len(labels)):
        seq = onehot_to_seq(inputs[i, :4, :])

        # Top positions by |SHAP| in 3'UTR
        utr_scores = nuc_shap_utr[i]
        top_utr_idx = np.argsort(utr_scores)[-n_top:]  # relative to utr_start
        top_positions = top_utr_idx + utr_start  # absolute positions

        for pos in top_positions:
            lo = pos - half_k
            hi = pos + half_k
            if lo < 0 or hi > W:
                continue
            kmer = seq[lo:hi]
            if "N" in kmer:
                continue
            if nmd[i]:
                nmd_high_shap_kmers[kmer] += 1
            else:
                ctrl_high_shap_kmers[kmer] += 1

        # All k-mers in 3'UTR (background)
        for pos in range(utr_start + half_k, min(W - half_k, utr_start + utr_len)):
            kmer = seq[pos - half_k:pos + half_k]
            if "N" in kmer:
                continue
            if nmd[i]:
                nmd_all_kmers[kmer] += 1
            else:
                ctrl_all_kmers[kmer] += 1

    # Compute enrichment: frequency in high-SHAP vs background
    all_kmers = set(nmd_high_shap_kmers.keys()) | set(ctrl_high_shap_kmers.keys())
    print(f"  Unique k-mers in high-SHAP positions: {len(all_kmers)}")

    nmd_high_total = sum(nmd_high_shap_kmers.values())
    ctrl_high_total = sum(ctrl_high_shap_kmers.values())
    nmd_bg_total = sum(nmd_all_kmers.values())
    ctrl_bg_total = sum(ctrl_all_kmers.values())

    rows = []
    for kmer in all_kmers:
        nmd_high = nmd_high_shap_kmers.get(kmer, 0)
        ctrl_high = ctrl_high_shap_kmers.get(kmer, 0)
        nmd_bg = nmd_all_kmers.get(kmer, 0)
        ctrl_bg = ctrl_all_kmers.get(kmer, 0)

        # Frequency in high-SHAP positions
        nmd_high_freq = nmd_high / nmd_high_total if nmd_high_total > 0 else 0
        ctrl_high_freq = ctrl_high / ctrl_high_total if ctrl_high_total > 0 else 0

        # Background frequency
        nmd_bg_freq = nmd_bg / nmd_bg_total if nmd_bg_total > 0 else 0
        ctrl_bg_freq = ctrl_bg / ctrl_bg_total if ctrl_bg_total > 0 else 0

        # Enrichment: high-SHAP freq / background freq
        nmd_enrichment = (nmd_high_freq / nmd_bg_freq) if nmd_bg_freq > 0 else 0
        ctrl_enrichment = (ctrl_high_freq / ctrl_bg_freq) if ctrl_bg_freq > 0 else 0

        # NMD-specificity: difference in enrichment
        nmd_specificity = nmd_enrichment - ctrl_enrichment

        rows.append({
            "kmer": kmer,
            "nmd_high_count": nmd_high,
            "ctrl_high_count": ctrl_high,
            "nmd_bg_count": nmd_bg,
            "ctrl_bg_count": ctrl_bg,
            "nmd_high_freq": nmd_high_freq,
            "ctrl_high_freq": ctrl_high_freq,
            "nmd_bg_freq": nmd_bg_freq,
            "ctrl_bg_freq": ctrl_bg_freq,
            "nmd_enrichment": nmd_enrichment,
            "ctrl_enrichment": ctrl_enrichment,
            "nmd_specificity": nmd_specificity,
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("  No k-mers found")
        return None

    df = df.sort_values("nmd_specificity", ascending=False)

    path = out_dir / f"motif_kmer_enrichment_{tag}.tsv"
    df.to_csv(path, sep="\t", index=False)
    print(f"  -> {path} ({len(df)} k-mers)")

    # Print top NMD-enriched and top control-enriched k-mers
    print(f"\n  Top 15 NMD-enriched k-mers (high |SHAP| positions, 3'UTR):")
    print(f"  {'k-mer':<12} {'NMD_enr':>8} {'Ctrl_enr':>8} {'NMD_spec':>9} {'NMD_n':>6} {'Ctrl_n':>6}")
    for _, row in df.head(15).iterrows():
        print(f"  {row['kmer']:<12} {row['nmd_enrichment']:>8.2f} {row['ctrl_enrichment']:>8.2f} "
              f"{row['nmd_specificity']:>+9.2f} {row['nmd_high_count']:>6.0f} {row['ctrl_high_count']:>6.0f}")

    print(f"\n  Top 15 Control-enriched k-mers:")
    for _, row in df.tail(15).iloc[::-1].iterrows():
        print(f"  {row['kmer']:<12} {row['nmd_enrichment']:>8.2f} {row['ctrl_enrichment']:>8.2f} "
              f"{row['nmd_specificity']:>+9.2f} {row['nmd_high_count']:>6.0f} {row['ctrl_high_count']:>6.0f}")

    return df


def approach_b_positional_kmer(shap, inputs, labels, channels, stop_pos,
                                out_dir, tag, k=6):
    """
    Positional k-mer analysis: for each position relative to stop codon,
    what k-mer has the highest mean |SHAP| in NMD vs control?
    """
    print(f"\n=== Approach B (positional): per-position top k-mers (k={k}) ===")
    nmd = labels == 1
    ctrl = labels == 0
    W = shap.shape[2]
    half_k = k // 2

    rows = []
    for pos in range(half_k, W - half_k, 3):  # every 3rd position for speed
        rel = pos - stop_pos

        # Extract k-mers and their SHAP at this position
        nmd_kmer_shap = {}
        ctrl_kmer_shap = {}

        for i in range(len(labels)):
            kmer = onehot_to_seq(inputs[i, :4, pos - half_k:pos + half_k])
            if "N" in kmer:
                continue
            total_shap = float(np.abs(shap[i, :4, pos]).sum())

            if nmd[i]:
                if kmer not in nmd_kmer_shap:
                    nmd_kmer_shap[kmer] = []
                nmd_kmer_shap[kmer].append(total_shap)
            else:
                if kmer not in ctrl_kmer_shap:
                    ctrl_kmer_shap[kmer] = []
                ctrl_kmer_shap[kmer].append(total_shap)

        # Find top k-mer by mean |SHAP| for NMD
        if nmd_kmer_shap:
            top_nmd = max(nmd_kmer_shap.items(),
                          key=lambda x: np.mean(x[1]) if len(x[1]) >= 5 else 0)
            top_nmd_kmer, top_nmd_shaps = top_nmd
        else:
            top_nmd_kmer, top_nmd_shaps = "N" * k, [0]

        if ctrl_kmer_shap:
            top_ctrl = max(ctrl_kmer_shap.items(),
                           key=lambda x: np.mean(x[1]) if len(x[1]) >= 5 else 0)
            top_ctrl_kmer, top_ctrl_shaps = top_ctrl
        else:
            top_ctrl_kmer, top_ctrl_shaps = "N" * k, [0]

        rows.append({
            "position": pos,
            "relative_position": rel,
            "top_nmd_kmer": top_nmd_kmer,
            "top_nmd_mean_shap": float(np.mean(top_nmd_shaps)),
            "top_nmd_count": len(top_nmd_shaps),
            "top_ctrl_kmer": top_ctrl_kmer,
            "top_ctrl_mean_shap": float(np.mean(top_ctrl_shaps)),
            "top_ctrl_count": len(top_ctrl_shaps),
            "n_unique_kmers_nmd": len(nmd_kmer_shap),
            "n_unique_kmers_ctrl": len(ctrl_kmer_shap),
        })

    df = pd.DataFrame(rows)
    path = out_dir / f"motif_positional_kmer_{tag}.tsv"
    df.to_csv(path, sep="\t", index=False)
    print(f"  -> {path} ({len(df)} positions)")

    return df


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="atg100_stop500")
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--k", type=int, default=8, help="k-mer size for approach B")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    tag = args.tag
    run_suffix = f"_run{args.run_id}" if args.run_id else ""
    file_tag = f"{tag}{run_suffix}"

    # Load ATG branch
    atg_path = results_dir / f"deepshap_atg_{file_tag}.npz"
    stop_path = results_dir / f"deepshap_stop_{file_tag}.npz"

    if not atg_path.exists() or not stop_path.exists():
        print(f"NPZ files not found for {file_tag}")
        return

    print(f"Motif analysis for {file_tag}")

    # ATG branch — full logo
    atg_shap, atg_inp, labels, atg_channels = load_npz(atg_path)
    nmd_n = (labels == 1).sum()
    ctrl_n = (labels == 0).sum()
    print(f"Samples: {len(labels)} (NMD={nmd_n}, ctrl={ctrl_n})")
    print(f"ATG shape: {atg_shap.shape}")

    approach_a_atg_logo(atg_shap, atg_inp, labels, atg_channels, results_dir, file_tag)

    # STOP branch
    stop_shap, stop_inp, labels_s, stop_channels = load_npz(stop_path)
    print(f"STOP shape: {stop_shap.shape}")

    stop_pos = find_stop_position(stop_inp, stop_channels)
    print(f"Stop codon at position {stop_pos}")

    # Approach A: logos at landmarks
    approach_a_logos(stop_shap, stop_inp, labels_s, stop_channels, stop_pos,
                     results_dir, file_tag)

    # Approach B: k-mer enrichment
    approach_b_kmer_enrichment(stop_shap, stop_inp, labels_s, stop_channels,
                                stop_pos, results_dir, file_tag, k=args.k)

    # Approach B (positional): per-position top k-mers
    approach_b_positional_kmer(stop_shap, stop_inp, labels_s, stop_channels,
                                stop_pos, results_dir, file_tag, k=6)

    print("\nDone.")


if __name__ == "__main__":
    main()

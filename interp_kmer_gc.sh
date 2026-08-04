#!/bin/bash
#SBATCH --job-name=hi_kmergc
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/p.castaldi/cc/nmd_orf_model_v5_4ct/results_ism_v6/interp_kmer_gc_%j.log
set -euo pipefail
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
S=analysis_plans/analysis_ism_regions.py
echo "script sha256: $(sha256sum $S)"
BANKS=$(ls -1 results_ism_v6/bank_interp_s{100,200,300,400,500}.h5 | paste -sd, -)
# THE CONTROL THE k=5 RESULT NEEDS. Top enriched 5-mers are U/A-rich and the
# depleted end is GC-rich, which is exactly what a composition-driven signal looks
# like rather than a motif. Rescoring by the ONE GC-preserving substitution holds
# channel 5 bitwise fixed. If the AU-rich enrichment survives it is sequence
# identity; if it collapses it was composition.
echo "### k=5, decay, GC-PRESERVING SUBSTITUTIONS ONLY"
$PY -W ignore $S --shards "$BANKS" --column dec --kmer 5 --gc-neutral-only

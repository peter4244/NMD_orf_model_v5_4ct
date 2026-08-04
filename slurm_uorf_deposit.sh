#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=uorf_dep
#SBATCH --output=results_deposit_2026-08-04/uorf_deposit_%j.log

# uORF attention inference under the DEPOSITED checkpoint (2026-08-04).
#
# tx_summary.tsv and nmd_orf_data.h5 are SYMLINKED from results_4ct_dn, on the same grounds
# as selected_orfs.tsv: both are DATA-derived (export_rds.R writes tx_summary, data_prep.py
# writes the H5), not model-derived, and the correct copies are the ones built in the same
# 2026-07-30 pass that produced the H5 these predictions were scored over. Symlinks rather
# than copies so the provenance is readable from an ls.
#
# The only model-derived file in this directory is the deposited checkpoint itself.

set -uo pipefail
cd ~/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
$PY infer_uorf_attention.py --results-dir results_deposit_2026-08-04
echo "EXIT_uorf=$?"
ls -la results_deposit_2026-08-04

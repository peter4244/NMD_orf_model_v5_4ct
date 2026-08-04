#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=exp_dep
#SBATCH --output=results_deposit_2026-08-04/exports_deposit_%j.log

# The two Python section-5 exports, run against the DEPOSITED checkpoint (2026-08-04).
#
# BOTH OF THESE WERE OOM-KILLED ON THE LOGIN NODE (137 and a bare Killed), which is the
# hazard this repo already records for build_h5. 64G is set against selected_orfs.tsv at
# 24 MB plus the predictions table plus whatever the loader materialises.
#
# selected_orfs.tsv is SYMLINKED from results_4ct_dn, deliberately and not copied. It is
# written by data_prep.py, so it is DATA-derived, not model-derived, and the correct copy to
# pair with these predictions is the one built beside the very H5 they were scored over
# (2026-07-30). The symlink keeps that provenance visible instead of burying it in a copy.
#
# NOT run here: the two R steps downstream. R on this cluster has 181 packages and not the
# ones the analysis needs (W292), so they run on the laptop against these outputs.

set -uo pipefail
cd ~/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
RES=results_deposit_2026-08-04

echo "########## infer_uorf_attention.py"
$PY infer_uorf_attention.py --results-dir $RES
echo "EXIT_infer=$?"

echo "########## 10_export_stop_codon_freq_sf37.py"
$PY 10_export_stop_codon_freq_sf37.py --results-dir $RES --tag atg500_stop500 \
    --split all --config config_dn.yaml
echo "EXIT_sf37=$?"

echo "########## outputs"
ls -la $RES

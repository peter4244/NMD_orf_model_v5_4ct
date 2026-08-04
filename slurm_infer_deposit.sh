#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=infer_dep
#SBATCH --output=results_deposit_2026-08-04/infer_deposit_%j.log

# Score every isoform with the DEPOSITED checkpoint (2026-08-04).
#
# WHY A JOB AND NOT THE LOGIN NODE: run directly on the login node this was OOM-killed in
# under 20 seconds, which is the hazard the repo already records for build_h5. --mem=32G is
# set against the 2.9 GB H5 plus the window tensors the loader materialises per batch.
#
# WHICH CHECKPOINT, AND WHY IT MATTERS: results_deposit_2026-08-04/ holds ONLY
# best_model_atg500_stop500.pt copied from the Zenodo deposit (sha256 dfd2ffca...), kept in
# its own directory so it can never be confused with the seed42 retrain in results_4ct_dn.
# The H5 comes from config_dn.yaml and is the deposit-native one.
#
# Output lands as predictions_atg500_stop500_all.tsv -- member_tag(tag, seed) + "_" + split,
# the convention evaluate.py already uses and that the four export scripts read. See W295.

set -euo pipefail
cd ~/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
$PY run_infer_all.py --config config_dn.yaml --atg-window 500 --stop-window 500 \
    --results-dir results_deposit_2026-08-04 --split all
echo "EXIT=$?"

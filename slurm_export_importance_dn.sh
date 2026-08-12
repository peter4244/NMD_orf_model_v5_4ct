#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=export_importance_dn
#SBATCH --output=results_deposit_h5_2026-08-04/export_importance_dn_%j.log

# DEPOSIT-NATIVE port of drivers/slurm_export_importance_v5.sh, 2026-08-12.
# The original reproduces the superseded sweep-era run: it targets results_4ct and
# config.yaml, whose selection was atg500_stop500. This one targets the deposit-native
# tree and reads the current selection from config_dn.yaml.

set -o pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
PY="${PY:-/home/p.castaldi/.conda/envs/nmd_model/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-results_deposit_h5_2026-08-04}"
# The window and tag come from the config, never from literals here.
TAG=$($PY paths_config.py --selected-tag --config config_dn.yaml) || { echo "cannot read selected tag" >&2; exit 1; }
ATG=${TAG#atg}; ATG=${ATG%%_*}; STOP=${TAG##*stop}
echo "=== node $(hostname) | tag=$TAG (atg=$ATG stop=$STOP) | results=$RESULTS_DIR ==="

overall=0
$PY 05_export_sample_importance.py --results-dir "$RESULTS_DIR" --tag "$TAG" \
    --atg-window "$ATG" --stop-window "$STOP"
rc=$?; echo "=== 05 exit: $rc ==="; [ $rc -ne 0 ] && overall=$rc
$PY 05b_export_sample_importance_tsv.py --results-dir "$RESULTS_DIR" --tag "$TAG"
rc=$?; echo "=== 05b exit: $rc ==="; [ $rc -ne 0 ] && overall=$rc
exit $overall

echo "=== done ==="

#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=fshap_dn
#SBATCH --output=results_deposit_h5_2026-08-04/fshap_dn_%j.log

# DEPOSIT-NATIVE port of drivers/slurm_fshap_all50.sh, 2026-08-12.
# The original reproduces the superseded sweep-era run: it targets results_4ct and
# config.yaml, whose selection was atg500_stop500. This one targets the deposit-native
# tree and reads the current selection from config_dn.yaml.

set -o pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
# PY resolves in three steps so it works off this machine without changing behaviour on it:
# the authoring env if present, else whatever python3 is on PATH, else a loud failure. It
# previously defaulted to the authoring path unconditionally, which resolves for one account
# and silently points everyone else at a path that does not exist.
PY="${PY:-/home/p.castaldi/.conda/envs/nmd_model/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3 || true)"
[ -x "$PY" ] || { echo "FATAL: no python found. Set PY to one with torch and shap installed "\
                       "(see environment-model.yml)." >&2; exit 1; }
RESULTS_DIR="${RESULTS_DIR:-results_deposit_h5_2026-08-04}"
# The window and tag come from the config, never from literals here.
TAG=$($PY paths_config.py --selected-tag --config config_dn.yaml) || { echo "cannot read selected tag" >&2; exit 1; }
ATG=${TAG#atg}; ATG=${ATG%%_*}; STOP=${TAG##*stop}
echo "=== node $(hostname) | tag=$TAG (atg=$ATG stop=$STOP) | results=$RESULTS_DIR ==="

$PY 12_feature_shap_structural.py --config config_dn.yaml --results-dir "$RESULTS_DIR" --tag "$TAG"
rc=$?; echo "=== 12 exit: $rc ==="; exit $rc

echo "=== done ==="

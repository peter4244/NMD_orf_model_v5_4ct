#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=interpret_dn
#SBATCH --output=results_deposit_h5_2026-08-04/interpret_dn_%j.log

# DEPOSIT-NATIVE port of drivers/slurm_interpret_v5.sh, 2026-08-12.
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
# The deposited member, and the split its inputs were scored on. paths_config.py
# --selected-tag deliberately returns the tag WITHOUT the seed, so the seed is named here.
MEMBER_SEED="${MEMBER_SEED:-42}"
SPLIT="${SPLIT:-test_clean}"
echo "=== node $(hostname) | tag=$TAG (atg=$ATG stop=$STOP) | results=$RESULTS_DIR ==="

overall=0
# --member-seed AND --split, because evaluate.py wrote {tag}_seed42_{split}. Passing --tag alone
# made 04 look for attention_weights_atg1000_stop1000.tsv, which nobody has ever written, so all
# four of its outputs were silently absent from the tree (W426).
$PY 04_interpret_attention.py --results-dir "$RESULTS_DIR" --tag "$TAG" \
    --member-seed "$MEMBER_SEED" --split "$SPLIT"
rc=$?; echo "=== 04 exit: $rc ==="; [ $rc -ne 0 ] && overall=$rc
$PY 05_interpret_structural.py --config config_dn.yaml --tag "$TAG" \
    --results-dir "$RESULTS_DIR" --atg-window "$ATG" --stop-window "$STOP" \
    --member-seed "$MEMBER_SEED"
rc=$?; echo "=== 05 exit: $rc ==="; [ $rc -ne 0 ] && overall=$rc
exit $overall

echo "=== done ==="

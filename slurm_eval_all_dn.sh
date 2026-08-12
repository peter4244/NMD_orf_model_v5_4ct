#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=dn_evalall
#SBATCH --output=results_deposit_h5_2026-08-04/eval_all_%j.log
#
# FULL-COHORT inference for the interpretation chain (D74/D77). Writes
# predictions_{tag}_seed{N}_all.tsv, which 08, 09b and 09_export_polya read to attach a label to
# every isoform.
#
# --full-cohort IS THE AFFIRMATION, not a formality. enforce_split_gate refuses --split all
# without it, because pooling training and held-out data is legitimate for interpretation and
# never for a performance number. The metrics JSON records evaluation_class=full_cohort and names
# its AUC auc_mixed_in_sample, so a consumer cannot read this as generalization. The TEST split is
# untouched here and is scored ONCE, at the end, with --final.
set -euo pipefail
# PY resolves in three steps so it works off this machine without changing behaviour on it:
# the authoring env if present, else whatever python3 is on PATH, else a loud failure. It
# previously defaulted to the authoring path unconditionally, which resolves for one account
# and silently points everyone else at a path that does not exist.
PY="${PY:-/home/p.castaldi/.conda/envs/nmd_model/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3 || true)"
[ -x "$PY" ] || { echo "FATAL: no python found. Set PY to one with torch and shap installed "\
                       "(see environment-model.yml)." >&2; exit 1; }
RESULTS_DIR="${RESULTS_DIR:-results_deposit_h5_2026-08-04}"
# Windows and tag come from the config, never from literals here -- 39 such literals
# across 26 drivers would otherwise keep reading 500/500 after a re-selection.
TAG=$($PY paths_config.py --selected-tag --config config_dn.yaml) || exit 1
ATG=${TAG#atg}; ATG=${ATG%%_*}; STOP=${TAG##*stop}
cd "$SLURM_SUBMIT_DIR"
echo "=== node $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
$PY evaluate.py --config config_dn.yaml --results-dir ${RESULTS_DIR:-results_deposit_h5_2026-08-04} \
    --atg-window "$ATG" --stop-window "$STOP" --member-seed 42 --split all --full-cohort
rc=$?
echo "=== evaluate exit: $rc ==="
exit $rc

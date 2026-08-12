#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=dn_uorf
#SBATCH --output=results_deposit_h5_2026-08-04/uorf_dn_%j.log
#
# uORF attention inference on the POST-CLIP model, for claim 5.4.3.
#
# SCOPE IS ALREADY CORRECT AND IS NOT CHANGED HERE. infer_uorf_attention.py runs over the full
# labeled universe (train+val+test) by construction -- it keeps every isoform present in
# tx_summary and applies no split filter -- and its docstring gives the reason: attention is an
# ATTRIBUTION analysis, so held-out discipline applies to AUC and not to where attention lands.
# That matches D74/D77. What changed is the MODEL, so this re-runs; the population does not move.
#
# DERIVED FROM provenance/slurm_uorf_and_inferall_dn.sh WITH TWO CHANGES:
#  1. the checkpoint guard tested best_model_{tag}.pt, which 03_train.py has not written since
#     members gained seeds. Under `set -e` that guard exited the job before any work. Fourth
#     driver carrying that assumption, after the trainer and the two SHAP drivers.
#  2. its other half re-ran run_infer_all.py to write predictions_all_{tag}.tsv. Dropped: evaluate
#     --split all --full-cohort already produced predictions_{tag}_seed42_all.tsv, and two files
#     holding one quantity under different names is the defect this project keeps paying for.
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
CKPT=${RESULTS_DIR:-results_deposit_h5_2026-08-04}/best_model_${TAG}_seed42.pt
test -e "$CKPT" || { echo "FATAL: $CKPT missing"; exit 1; }
if cmp -s "$CKPT" results_4ct/best_model_atg500_stop500.pt; then
  echo "FATAL: dn checkpoint is byte-identical to the published one -- wrong tree"; exit 1
fi
echo "checkpoint present and differs from published: OK"
NMD_RESULTS_DIR=${RESULTS_DIR:-results_deposit_h5_2026-08-04} $PY infer_uorf_attention.py --config config_dn.yaml --results-dir ${RESULTS_DIR:-results_deposit_h5_2026-08-04} --member-seed 42
rc=$?
echo "=== infer_uorf_attention exit: $rc ==="
exit $rc

# THIS WRAPPER RUNS INFERENCE ONLY. compute_uorf_attention_metrics.R consumes what it writes and
# is a separate step -- REPRODUCTION.md described this job as doing both, which left the metrics
# unbuilt while every command reported success. Run it after this job completes:
#   Rscript compute_uorf_attention_metrics.R --results-dir "$RESULTS_DIR"

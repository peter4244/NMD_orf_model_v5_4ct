#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=dn_uorf
#SBATCH --output=results_4ct_dn/uorf_dn_%j.log
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
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
cd "$SLURM_SUBMIT_DIR"
echo "=== node $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
CKPT=results_4ct_dn/best_model_atg500_stop500_seed42.pt
test -e "$CKPT" || { echo "FATAL: $CKPT missing"; exit 1; }
if cmp -s "$CKPT" results_4ct/best_model_atg500_stop500.pt; then
  echo "FATAL: dn checkpoint is byte-identical to the published one -- wrong tree"; exit 1
fi
echo "checkpoint present and differs from published: OK"
NMD_RESULTS_DIR=results_4ct_dn $PY infer_uorf_attention.py --results-dir results_4ct_dn --member-seed 42
rc=$?
echo "=== infer_uorf_attention exit: $rc ==="
exit $rc

#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=dn_inferall_uorf
#SBATCH --output=results_4ct_dn/inferall_uorf_dn_%j.log

# The two deposit-native GPU artifacts the deposit ships but the dn run never produced:
#   predictions_all_atg500_stop500.tsv   -- full-cohort inference (section 5 predictor benchmark)
#   uorf_attention_predictions.tsv       -- input to the uORF attention metrics
#
# BOTH DEFAULT TO THE PUBLISHED TREE. run_infer_all.py takes --results-dir; infer_uorf_attention.py
# did NOT until today -- it hardcoded CKPT, H5, TXSUM *and* OUT to results_4ct, so run unchanged
# against a deposit-native question it would have loaded the April checkpoint and then overwritten
# the published output it is compared against. Sixth instance of that defect in this repo
# (03_train, evaluate, 11_kernel_shap_branches, 09_export_gc_content, 09_export_junction_ordinal
# were the first five). It now honours NMD_RESULTS_DIR, default unchanged.
set -euo pipefail
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
cd "$SLURM_SUBMIT_DIR"

echo "=== node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Guard: refuse to run if the deposit-native checkpoint is missing, or is byte-identical to the
# published one -- that would mean we are pointed at the wrong tree and the whole run is void.
test -e results_4ct_dn/best_model_atg500_stop500.pt
if cmp -s results_4ct_dn/best_model_atg500_stop500.pt results_4ct/best_model_atg500_stop500.pt; then
  echo "FATAL: dn checkpoint is byte-identical to the published one -- wrong tree"; exit 1
fi
echo "checkpoint differs from published: OK"

echo
echo "=== [1/2] full-cohort inference -> predictions_all_atg500_stop500.tsv ==="
$PY run_infer_all.py --results-dir results_4ct_dn --split all --atg-window 500 --stop-window 500
echo "  [1/2] exit=$?"

echo
echo "=== [2/2] uORF attention inference -> uorf_attention_predictions.tsv ==="
NMD_RESULTS_DIR=results_4ct_dn $PY infer_uorf_attention.py
echo "  [2/2] exit=$?"

echo
echo "=== produced ==="
ls -la results_4ct_dn/predictions_all_atg500_stop500.tsv results_4ct_dn/uorf_attention_predictions.tsv

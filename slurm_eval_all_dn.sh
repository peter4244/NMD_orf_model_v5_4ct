#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=dn_evalall
#SBATCH --output=results_4ct_dn/eval_all_%j.log
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
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
cd "$SLURM_SUBMIT_DIR"
echo "=== node $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
$PY evaluate.py --config config_dn.yaml --results-dir results_4ct_dn \
    --atg-window 500 --stop-window 500 --member-seed 42 --split all --full-cohort
rc=$?
echo "=== evaluate exit: $rc ==="
exit $rc

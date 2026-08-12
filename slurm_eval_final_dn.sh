#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=dn_final
#SBATCH --output=results_deposit_h5_2026-08-04/eval_final_%j.log
#
# THE ONE-SHOT HELD-OUT EVALUATION. Claims 5.2.1 and 5.6.4 -- the only two section 5 numbers that
# are test-only. Everything interpretive is full cohort (D74/D77); performance is not.
#
# RUN THIS ONCE, AND LAST. The test split has been untouched through the entire regeneration:
# training early-stopped on val_clean, the interpretation chain ran on --split all --full-cohort,
# and every export read the pooled predictions. --final is the affirmation that this is the single
# evaluation allowed to touch chr1/3/5/7, and the metrics JSON records evaluation_class=final_test
# so no later reader can mistake a development number for this one.
#
# test_clean IS THE PARALOG-FREE TEST SET, which is what the manuscript describes as "held-out
# chr 1/3/5/7 paralog-free" and what the published run used -- evaluate.py hardcoded test_clean
# before it took a --split at all. _split_mask defines it as splits == "test", with the 122
# paralog-straddling transcripts carried under their own test_paralog label rather than filtered
# here. Naming the split by its meaning, not reconstructing it, is why this cannot drift.
set -euo pipefail
PY="${PY:-/home/p.castaldi/.conda/envs/nmd_model/bin/python}"
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
    --atg-window "$ATG" --stop-window "$STOP" --member-seed 42 --split test_clean --final
rc=$?
echo "=== evaluate exit: $rc ==="
exit $rc

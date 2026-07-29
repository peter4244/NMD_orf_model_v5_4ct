#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1   # UNPINNED 2026-07-29 (reversing the pin earlier the same day).
#   Pinning to v100-sxm2 looked like a 34->14 node trade; it is not. sinfo -N emits one row
#   per node-PARTITION pair, so that 14 was duplicates -- and d1002/d1007/d1009/d1010 are
#   DRAINING, leaving ~2 nodes with capacity. Pinned, job 8826324 projected a start 2h LATER
#   than the oversized job it replaced. Eligibility, not gap size, is the binding constraint.
#   For SHORT work unpinning is free: the wall below covers the ~5x slow draw. Long jobs
#   (joint/atg-stop/all-modes) stay pinned, since unpinned they would need 2-6h walls.
#SBATCH --time=01:00:00   # 8:08 on the fast family, ~41m on the slow one; this
#   wall survives EITHER draw, which is what makes unpinning safe.
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=dn_train
#SBATCH --output=results_4ct_dn/train_dn_%j.log

# DEPOSIT-NATIVE RETRAIN, ATG=500 STOP=500 (2026-07-27).
#
# Everything it reads was regenerated from the Zenodo deposit:
#   universe   42,043 isoforms (published 39,938), labels 9,425/32,618 = 1:3.46
#   splits     deterministic by chromosome -- test chr1/3/5/7, val chr2/4
#   sequences  proven byte-identical to the deposit FASTA for all 42,043 (digest
#              16b3d11d7755c3338021c350f34732f8 on both sides)
#
# --results-dir results_4ct_dn keeps the PUBLISHED results_4ct intact; the whole point is
# to compare against it, and 03_train.py used to hardcode the path and would have
# overwritten the checkpoint and metrics it is being compared with.
#
# Determinism was verified on a V100 at THIS window config before this job existed
# (verify_determinism.py --atg 500 --stop 500: two seeded runs bit-identical).
# No `set -e`: a non-zero exit is a RESULT to read in the log, not something to hide.
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
echo "=== node $(hostname) ==="; nvidia-smi --query-gpu=name --format=csv,noheader

echo "=== TRAIN (deposit-native) ==="
$PY 03_train.py --config config_dn.yaml --results-dir results_4ct_dn --atg-window 500 --stop-window 500
echo "=== train exit: $? ==="

echo "=== EVALUATE ==="
# --split IS NOW REQUIRED (2026-07-29). evaluate.py used to hardcode split="test_clean",
# which is why every metrics_*.json in existence is a test-set score -- including the twelve
# that selected the published window config. Routine monitoring scores VAL; the test set is
# a deliberate one-time act needing --final, shown commented below so it cannot happen by
# habit.
$PY evaluate.py --config config_dn.yaml --results-dir results_4ct_dn --atg-window 500 --stop-window 500 --split val
# FINAL evaluation, run deliberately and once the config is settled:
# $PY evaluate.py --config config_dn.yaml --results-dir results_4ct_dn --atg-window 500 --stop-window 500 --split test_clean --final
echo "=== evaluate exit: $? ==="

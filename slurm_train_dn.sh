#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
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
$PY evaluate.py --config config_dn.yaml --results-dir results_4ct_dn --atg-window 500 --stop-window 500
echo "=== evaluate exit: $? ==="

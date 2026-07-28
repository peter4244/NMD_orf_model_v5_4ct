#!/bin/bash
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=dn_det
#SBATCH --output=results_4ct_dn/determinism_%j.log

# Must PASS before 03_train.py. A CPU pass says nothing about the GPU path, and
# use_deterministic_algorithms(True) raises if any op in THIS architecture lacks a
# deterministic kernel -- 30 seconds here beats discovering it hours into training.
# No `set -e`: a non-zero exit from the harness is a RESULT to read, not a job to hide.
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null
$PY verify_determinism.py --steps 5 --atg 500 --stop 500
echo "=== verify_determinism exit: $? ==="

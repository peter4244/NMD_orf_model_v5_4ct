#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=04:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=shap_r5
#SBATCH --output=results_deposit_h5_2026-08-04/deepshap_run5_atgstop_%j.log

# The "atg stop" decomposition for replicate 5, which OOM-killed in array job 8941172_5 after
# joint, structural and atg had all completed. 19 of the 20 npz files exist; this writes the
# 20th.
#
# --mem RAISED 32G -> 96G, and the array request was wrong for EVERY replicate, not only the
# one that died. Measured MaxRSS across the five full-cohort tasks: 42.3, 42.3, 43.1, 49.8 and
# 49.5 GB, all against a 32G request. Four survived on node slack rather than by fitting, so
# whether a replicate completed was a property of the node it landed on. That is not a thin
# margin, it is a coin toss that came up heads four times.
#
# Rerunning the whole MODE rather than only the stop branch is deliberate. "atg stop" writes
# both npz files in one invocation, and DeepSHAP is nondeterministic here by design
# (NMD_ALLOW_NONDETERMINISM=1 is exactly why five replicates exist). A fresh pair drawn from
# one run is more coherent than a stop drawn hours apart from the atg it is paired with. The
# existing run5 atg file is overwritten, which is correct rather than lossy: it was one draw
# from a distribution and this is another.
export NMD_ALLOW_NONDETERMINISM=1
set -uo pipefail
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
RESULTS_DIR=results_deposit_h5_2026-08-04

# Windows from the config, never literals -- the selection moved to atg1000_stop1000.
TAG=$($PY paths_config.py --selected-tag --config config_dn.yaml) || exit 1
ATG=${TAG#atg}; ATG=${ATG%%_*}; STOP=${TAG##*stop}
echo "=== node $(hostname) | run 5 | tag=$TAG ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null

$PY deepshap.py --config config_dn.yaml --results-dir "$RESULTS_DIR" \
    --n-explain 0 --n-background 500 --atg-window "$ATG" --stop-window "$STOP" \
    --seed 500 --run-id 5 --member-seed 42 --split all --full-cohort --branches atg stop
RC=$?

# The exit status is the Python process, not the echo. Job 8941289 reported COMPLETED 0:0 on a
# KeyError because its last statement was an echo.
echo "EXIT_atgstop=$RC"
exit $RC

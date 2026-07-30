#!/bin/bash
#SBATCH --partition=sharing
#SBATCH --time=00:55:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ds_interp
#SBATCH --output=results_4ct_interp/ds_%j.log

# DeepSHAP for ONE member of ONE configuration, for the v3 interpretation plan.
#
# A BATCH JOB, NOT A LOGIN-NODE RUN. ensemble_evaluate.py was run over ssh on 2026-07-30 and died
# silently -- no output, no files, no traceback -- because the login node killed it. This cluster is
# documented as OOM-killing login-node work (slurm_build_h5_dn.sh calls the conda python by absolute
# path for exactly that reason). Scoring a model over a split held in RAM as float32 is not
# login-node work, and DeepSHAP is heavier than scoring.
#
# Parameters arrive as environment variables, matching slurm_sweep_member.sh. Required: DS_ATG,
# DS_STOP, DS_SEED. Everything else has a plan-derived default.
#
#   DS_ATG=2000 DS_STOP=500 DS_SEED=100 sbatch slurm_deepshap_interp.sh
set -euo pipefail
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

: "${DS_ATG:?DS_ATG not set}"
: "${DS_STOP:?DS_STOP not set}"
: "${DS_SEED:?DS_SEED not set}"
DS_SPLIT="${DS_SPLIT:-all}"
DS_RUN="${DS_RUN:-1}"
DS_RESULTS="${DS_RESULTS:-results_4ct_interp}"
DS_NEXPLAIN="${DS_NEXPLAIN:-2000}"
DS_BRANCHES="${DS_BRANCHES:-atg stop}"

mkdir -p "$DS_RESULTS"

# --full-cohort is passed explicitly because DS_SPLIT defaults to `all`: the interpretation
# analyses run over the full labelled universe by project convention (only AUC/AUPRC are
# test-only). It is spelled out here rather than inherited, and if DS_SPLIT is ever set to a test
# split this will fail the gate rather than quietly scoring held-out data -- which is the point.
GATE=""
[ "$DS_SPLIT" = "all" ] && GATE="--full-cohort"

echo "=== node $(hostname) | deepshap atg${DS_ATG}/stop${DS_STOP} seed ${DS_SEED} | split ${DS_SPLIT} | run ${DS_RUN} ==="
$PY -V
$PY deepshap.py \
    --config config_dn.yaml \
    --results-dir "$DS_RESULTS" \
    --atg-window "$DS_ATG" \
    --stop-window "$DS_STOP" \
    --member-seed "$DS_SEED" \
    --split "$DS_SPLIT" \
    $GATE \
    --branches $DS_BRANCHES \
    --run-id "$DS_RUN" \
    --n-explain "$DS_NEXPLAIN"
echo "=== deepshap exit: $? ==="

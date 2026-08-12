#!/bin/bash
#SBATCH --partition=sharing
#SBATCH --time=00:40:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ens_eval
#SBATCH --output=slurm_logs/ens_eval_%j.log

# ENSEMBLE EVALUATION AS A BATCH JOB, because the first attempt died on the login node.
#
# 2026-07-30: ensemble_evaluate.py was run directly over ssh and produced NO output and NO files.
# Not a code fault -- it imports and parses fine -- the login node killed it. This cluster is
# already documented as OOM-killing login-node work: slurm_build_h5_dn.sh calls the conda python by
# ABSOLUTE PATH precisely because conda init was being OOM-killed in .bashrc there. Scoring five
# members over a split held in RAM as float32 is not login-node work.
#
# The silent death is the part worth guarding against rather than the memory: no traceback, no
# partial file, nothing in the log. A step that fails invisibly is worse than one that fails loudly,
# which is why PYTHONUNBUFFERED is set below for the same reason it was added to the sweep member.
#
#   sbatch slurm_ensemble_eval.sh                       # defaults to the primary model
#   ENS_ATG=2000 ENS_STOP=2000 sbatch slurm_ensemble_eval.sh
#   ENS_SPLIT=val_clean ENS_RESULTS=results_4ct_sweep sbatch slurm_ensemble_eval.sh
set -euo pipefail
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
export PYTHONUNBUFFERED=1

# Defaults are the PRIMARY model selected 2026-07-30 (atg2000_stop100), overridable per invocation
# so the secondary contrast does not need a second copy of this file.
ATG="${ENS_ATG:-2000}"
STOP="${ENS_STOP:-100}"
SPLIT="${ENS_SPLIT:-val_clean}"
RESULTS="${ENS_RESULTS:-results_4ct_sweep}"
SEEDS="${ENS_SEEDS:-100,200,300,400,500}"

# NO --final AND NO --full-cohort HERE, DELIBERATELY. Both gates exist because the published
# configuration was selected on twelve test-set scores. Passing either from a batch script is how an
# affirmation becomes a default that nobody re-reads; a test evaluation is one shot and should be
# typed deliberately, on its own decision, not inherited from a wrapper.
echo "=== node $(hostname) | ensemble ${ATG}/${STOP} | split ${SPLIT} | seeds ${SEEDS} ==="
$PY -V
$PY ensemble_evaluate.py \
    --config config_dn.yaml \
    --results-dir "${RESULTS}" \
    --atg-window "${ATG}" \
    --stop-window "${STOP}" \
    --seeds "${SEEDS}" \
    --split "${SPLIT}"
echo "=== ensemble_evaluate exit: $? ==="

#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=c6_regrammar
#SBATCH --output=analysis_plans/runlog_c6_%j.txt
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
export NMD_TOOLS=$HOME/cc/tools
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python3
echo "=== node $(hostname)  job ${SLURM_JOB_ID} ==="
for spec in "model_simpson_mechanism --all" "model_weight_decomposition --all" "model_regime_reconstruction"; do
  set -- $spec; f=$1; shift
  export NMD_CLAIM_VALUES=$PWD/analysis_plans/c6_${f}_${SLURM_JOB_ID}.tsv
  echo "--- $f $@ -> $NMD_CLAIM_VALUES"
  $PY analysis_plans/$f.py --out analysis_plans/c6_${f}_${SLURM_JOB_ID}.log --seed 100 "$@"
  echo "--- $f exit: $?"
done

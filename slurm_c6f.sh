#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=c6_final
#SBATCH --output=analysis_plans/runlog_c6f_%j.txt
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
export NMD_TOOLS=$HOME/cc/tools
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python3
J=${SLURM_JOB_ID}
echo "=== node $(hostname)  job $J ==="
run () {                      # $1 = tag, rest = producer + args
  local tag=$1; shift
  local prod=$1; shift
  export NMD_CLAIM_VALUES=$PWD/analysis_plans/f_${tag}_${J}.tsv
  $PY $NMD_TOOLS/trace_reads.py --out analysis_plans/f_${tag}_${J}.trace.tsv \
      --run analysis_plans/${prod}.py -- --out analysis_plans/f_${tag}_${J}.log "$@"
  echo "--- $tag exit: $?"
}
run encoder      model_encoder_channel_share
run geometry     model_scanning_geometry
run histograms   model_weight_histograms --all --seed 100
run tie          model_tie_estimator_check
run pool         model_pool_admission
run regimeall    model_regime_reconstruction --seed 100 --all

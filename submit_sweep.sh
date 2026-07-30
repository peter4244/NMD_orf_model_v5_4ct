#!/bin/bash
# submit_sweep.sh -- launch the window sweep as 60 INDIVIDUAL jobs, not an array.
#
# 12 configurations x 5 seeds. D31 re-runs the sweep on train+val only, and reverses D11's
# single seed ("five seeds is the point"), so the grid is two-dimensional.
#
# WHY 60 SEPARATE SUBMISSIONS RATHER THAN --array=1-60, WHICH IS THE IDIOMATIC ANSWER.
# Measured on this cluster 2026-07-29, as a controlled experiment: two probe jobs with
# IDENTICAL resource requests (partition=short, 32G, 16 cpus, 1h) submitted one second apart,
# differing only in task count.
#
#     single, 1 task   submit 20:20:29  ->  start 20:20:31   2 SECONDS
#     array,  4 tasks  submit 20:20:30  ->  still pending, projected 21:35   ~75 MINUTES
#
# So array-ness is the constraint here, not priority and not capacity. A 60-task array would
# have been far worse. This was worth measuring rather than assuming: the earlier attempt to
# run 5 members as --array=2-5 sat pending for the whole session while a lone job scheduled
# instantly.
#
# Individual jobs also fail independently, which matters for diagnosis: one bad cell does not
# hold up 59 others, and each has its own log named by its own job id.
#
#   ./submit_sweep.sh --dry-run     # print what would be submitted, submit nothing
#   ./submit_sweep.sh               # submit all 60
#   ./submit_sweep.sh --configs-only  # 12 jobs at one seed, for a cheap shakedown
#
# NOTE: as of 2026-07-29 every cell will FAIL IN ITS PREFLIGHT, on purpose, because the HDF5
# has no val_paralog label until W116 lands. That is the intended behaviour -- 60 fast, loud
# failures rather than 12 CPU-hours followed by 60 identical errors at the evaluate step.
# Run --dry-run until W116 is in.

set -euo pipefail
cd "$(dirname "$0")"

ATG_SIZES=(100 500 1000)
STOP_SIZES=(100 500 1000 2000)
SEEDS=(100 200 300 400 500)

DRY=0
ONE_SEED=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --configs-only) ONE_SEED=1 ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

[ "$ONE_SEED" = 1 ] && SEEDS=(100)

mkdir -p results_4ct_sweep
n=0
declare -a ids=()
for atg in "${ATG_SIZES[@]}"; do
  for stop in "${STOP_SIZES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      n=$((n + 1))
      name="sw_${atg}_${stop}_s${seed}"
      if [ "$DRY" = 1 ]; then
        printf "  [%2d] atg=%-4s stop=%-4s seed=%-3s  job-name=%s\n" "$n" "$atg" "$stop" "$seed" "$name"
      else
        id=$(sbatch --parsable --job-name="$name" \
               --export=ALL,SW_ATG="$atg",SW_STOP="$stop",SW_SEED="$seed" \
               slurm_sweep_member.sh)
        ids+=("$id")
        printf "  [%2d] atg=%-4s stop=%-4s seed=%-3s  -> %s\n" "$n" "$atg" "$stop" "$seed" "$id"
        # Space the submissions slightly: 60 sbatch calls in a burst can trip RPC rate limits.
        sleep 0.3
      fi
    done
  done
done

echo
if [ "$DRY" = 1 ]; then
  echo "DRY RUN: $n cells would be submitted as $n individual jobs (no array)."
else
  echo "submitted $n jobs"
  printf '%s\n' "${ids[@]}" > results_4ct_sweep/sweep_job_ids.txt
  echo "job ids -> results_4ct_sweep/sweep_job_ids.txt"
  echo "monitor : squeue -u \$USER -o '%.12i %.14j %.9T %.7M'"
  echo "collect : python3 collect_sweep.py"
fi

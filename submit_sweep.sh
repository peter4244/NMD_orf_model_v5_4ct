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
SEED_MAJOR=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --configs-only) ONE_SEED=1 ;;
    --seed-major) SEED_MAJOR=1 ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

[ "$ONE_SEED" = 1 ] && SEEDS=(100)

mkdir -p results_4ct_sweep

# ORDER MATTERS WHEN THROUGHPUT IS NEAR-SERIAL (2026-07-29). Measured: this cluster grants me
# roughly one slot at a time, so 60 cells run in ~3 hours essentially in submission order --
# and the order therefore decides WHEN each thing is learned.
#
# Config-major (the original nesting) completes all 5 seeds of one config before starting the
# next, so after 5 cells you know one configuration precisely and nothing about the other
# eleven. The first config in the grid is atg100_stop100, historically the WORST cell. Fifteen
# minutes bought the least useful fact available.
#
# --seed-major runs all 12 configs at seed 100, then all 12 at seed 200, and so on. After 12
# cells there is a complete 12-config ranking (comparable to the published single-seed sweep);
# after 24 the first spread estimate; and it sharpens from there.
#
# THIS IS FREE OF METHODOLOGICAL COST **ONLY BECAUSE ALL 60 STILL RUN.** Reordering changes when
# results appear, not which cells exist, so the final analysis is identical. It would stop being
# free the moment anyone stopped early on a favourable partial ranking -- that is selection on
# the outcome, and it needs a pre-registered rule (D34), not a scheduling decision.
build_cells() {
  if [ "$SEED_MAJOR" = 1 ]; then
    for seed in "${SEEDS[@]}"; do
      for atg in "${ATG_SIZES[@]}"; do
        for stop in "${STOP_SIZES[@]}"; do echo "$atg $stop $seed"; done
      done
    done
  else
    for atg in "${ATG_SIZES[@]}"; do
      for stop in "${STOP_SIZES[@]}"; do
        for seed in "${SEEDS[@]}"; do echo "$atg $stop $seed"; done
      done
    done
  fi
}

n=0; skipped=0
declare -a ids=()
while read -r atg stop seed; do
    do_cell() {
      n=$((n + 1))
      # RESUMABLE. Skip a cell whose metrics file already exists, so a reorder or a
      # resubmission after a partial run cannot duplicate finished work -- and cannot produce a
      # second copy of an output under the same name.
      if [ -f "results_4ct_sweep/metrics_atg${atg}_stop${stop}_seed${seed}_val_clean.json" ]; then
        skipped=$((skipped + 1)); return 0
      fi
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
    }
    do_cell
done < <(build_cells)

echo
if [ "$DRY" = 1 ]; then
  echo "DRY RUN: $((n - skipped)) cells would be submitted individually; $skipped already complete."
else
  echo "submitted $((n - skipped)) jobs; skipped $skipped already-complete cell(s)"
  printf '%s\n' "${ids[@]}" > results_4ct_sweep/sweep_job_ids.txt
  echo "job ids -> results_4ct_sweep/sweep_job_ids.txt"
  echo "monitor : squeue -u \$USER -o '%.12i %.14j %.9T %.7M'"
  echo "collect : python3 collect_sweep.py"
fi

#!/bin/bash
#SBATCH --partition=sharing
#SBATCH --time=00:50:00
#SBATCH --mem=40G
#
# MEMORY IS SET FOR THE WORST CELL, NOT THE AVERAGE (fixed 2026-07-29 after 3 OOM kills).
# I had cut this to 16G from a 7.86 GB measurement -- taken on atg500_stop500, one cell of a
# grid whose memory spans 20x. utils.py loads a whole split into RAM as float32, so the windows
# alone are 26,887 x 5 ORFs x 9 channels x W x 4 bytes, twice (ATG and stop):
#
#     ~4.84 GB per 1000 positions of window
#     atg500_stop500   ~9.7 GB    fits 16G
#     atg500_stop1000  ~14.5 GB   marginal
#     atg100_stop2000  ~20.4 GB   EXCEEDS 16G  <- killed, exit 0:9
#     atg1000_stop2000 ~14.5 GB train + val    <- worst case ~17-20 GB
#
# Jobs 8827523/8827525/8827526 died with SIGKILL mid-training on c3182, all large-window cells,
# while the SAME configs had succeeded earlier under the previous 32G request. 40G covers the
# worst cell with headroom; the cost of over-requesting on shared nodes is slower placement,
# and the cost of under-requesting is a dead job with no partial output.
#SBATCH --cpus-per-task=8
#
# MOVED OFF `short` TO `sharing`, 2026-07-29, measured not guessed. `short` is in bad shape --
# 91 nodes down*, 58 draining, 29 usable against 725 queued jobs, i.e. 25 pending per usable
# node. `sharing` had 123 usable nodes and 218 pending: 1.8 per node, ~14x better. Probe: three
# 8cpu/16G jobs submitted to `sharing` at 21:49:17 all STARTED at 21:49:46 -- 29 seconds, and
# concurrently -- while an identical job on `short` was still pending. On `short` the sweep was
# managing about two cells per 37 minutes, which for 53 remaining cells is most of a day.
#
# Request halved to 8 cpu / 16G because `sharing` nodes are shared: a smaller footprint places
# far more easily, and measured peak use is 7.9 GB. Fewer threads costs some per-cell time
# (3:19 at 16 threads), which is a trivial price for running several cells at once instead of
# queueing behind 725 jobs.
#
# 00:50:00 fits inside `sharing`'s 1-hour cap with room to spare: measured cell runtimes are
# 2:50 to 11:59, graded by window size -- stop=2000 moves 20x the data of stop=100, which is why
# a single "~3 min per cell" figure was misleading.
#SBATCH --job-name=sw
#SBATCH --output=results_sweep_dn_2026-08-04/sweep_%j.log
#
# ONE cell of the window sweep: one (atg, stop) configuration at one seed, on CPU.
# Submitted 60 times by submit_sweep.sh -- NOT as an array. See that script for why.
#
# Parameters arrive as environment variables because sbatch --export is how a single-task
# submission is parameterised. Required: SW_ATG, SW_STOP, SW_SEED.
#
# CPU, NOT GPU (measured 2026-07-29). 34,050 trainable parameters, dominated by moving
# 500-position windows through 32 conv channels rather than by arithmetic, so a V100 is
# largely idle: 12:20 wall on 16 CPU threads (job 8826508) against 8:08 on a V100 (job
# 8784272). 1.4x slower per run, and the `gpu` queue was projecting FOUR HOURS while `short`
# scheduled a single job in 2 seconds. For 60 runs that difference is the whole schedule.
#
# SCORED ON val_clean, per D31: the published configuration was selected on twelve TEST-set
# scores, and re-running on train+val is what repairs that. `--split val` would relocate the
# leak from test to validation rather than remove it, so it is not offered here.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-/home/p.castaldi/.conda/envs/nmd_model/bin/python}"

# UNBUFFERED, so a killed cell still leaves a diagnostic trail (2026-07-30).
# Jobs 8828520/21/22 were SIGKILLed at 8:42 and their logs ended at the "=== TRAIN ===" banner
# with NOT ONE line of Python output -- not the device, not the split sizes, not an epoch. That
# was stdout buffering, not the failure point: Python block-buffers when stdout is a file, and a
# SIGKILL discards the buffer. So the one artifact that would say how far the cell got, and
# whether it died during the RAM-hungry NMDDataset load, was destroyed by the kill itself.
# A crash you cannot diagnose costs more than the buffering saves.
export PYTHONUNBUFFERED=1

: "${SW_ATG:?SW_ATG not set -- submit via submit_sweep.sh}"
: "${SW_STOP:?SW_STOP not set}"
: "${SW_SEED:?SW_SEED not set}"

OUT=${SWEEP_OUT:-results_sweep_dn_2026-08-04}
mkdir -p "$OUT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=== node $(hostname) | CPU | atg=${SW_ATG} stop=${SW_STOP} seed=${SW_SEED} | threads=${OMP_NUM_THREADS}"

# PREFLIGHT. Seconds, and it runs BEFORE ~12 minutes of training.
#
# The val_paralog check is the one that matters and it is deliberately fatal. As of
# 2026-07-29 the HDF5 carries no val_paralog label, because 05u_paralog_annotation.R defines
# leakage only across the chr1/3/5/7 split (W116 extends it to the validation chromosomes).
# Until that lands, evaluate.py --split val_clean RAISES BY DESIGN, so that val_clean cannot
# quietly mean "unscreened". Without this preflight, launching the sweep early would burn
# 60 x 12 minutes of CPU and then fail 60 times at the evaluate step. With it, all 60 fail in
# seconds and say why.
$PY - <<'PRE'
import sys, yaml, h5py
cfg = yaml.safe_load(open("config_dn.yaml"))
h5 = cfg["data"]["hdf5_path"]
with h5py.File(h5, "r") as f:
    splits = {s.decode() if isinstance(s, bytes) else s for s in f["split"][:]}
if "val_paralog" not in splits:
    sys.exit(
        f"PREFLIGHT FAIL: {h5} has no 'val_paralog' label, so --split val_clean will raise.\n"
        f"  present: {sorted(splits)}\n"
        f"  The sweep must score on val_clean (D31). Land W116 -- extend 05u's leakage screen\n"
        f"  to VAL_CHRS, emitting a distinct val_leakage_genes -- then rebuild the HDF5.\n"
        f"  Do NOT substitute --split val: that selects on an unscreened validation set,\n"
        f"  which relocates the published leak instead of repairing it.")
print(f"  preflight ok: splits {sorted(splits)}")
PRE

TAG="atg${SW_ATG}_stop${SW_STOP}"
echo "=== TRAIN ${TAG} seed ${SW_SEED} ==="
$PY 03_train.py --config config_dn.yaml --results-dir "$OUT" \
    --atg-window "${SW_ATG}" --stop-window "${SW_STOP}" --seed "${SW_SEED}"
echo "=== train exit: $? ==="

echo "=== EVALUATE ${TAG} seed ${SW_SEED} on val_clean ==="
$PY evaluate.py --config config_dn.yaml --results-dir "$OUT" \
    --atg-window "${SW_ATG}" --stop-window "${SW_STOP}" \
    --member-seed "${SW_SEED}" --split val_clean
echo "=== evaluate exit: $? ==="

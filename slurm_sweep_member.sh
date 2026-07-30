#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=sw
#SBATCH --output=results_4ct_sweep/sweep_%j.log
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
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python

: "${SW_ATG:?SW_ATG not set -- submit via submit_sweep.sh}"
: "${SW_STOP:?SW_STOP not set}"
: "${SW_SEED:?SW_SEED not set}"

OUT=results_4ct_sweep
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

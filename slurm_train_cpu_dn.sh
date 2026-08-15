#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=dn_cpu
#SBATCH --output=results_4ct_dn_cpu/train_cpu_dn_%A_%a.log
#SBATCH --array=1
#
# Train one ensemble member on CPU. NO GPU REQUESTED.
#
# WHY THIS EXISTS. Measured 2026-07-29: this model does not need a GPU. It has 34,050
# trainable parameters, and the work is dominated by moving 500-position windows through 32
# conv channels rather than by arithmetic, so a V100 is largely idle. Timed on an 8-thread
# laptop CPU (under x86 emulation, so pessimistic): 2.5 ms/sample over 9 batches post-warmup,
# which extrapolates to ~1.1 min/epoch and ~7 min for a 6-epoch run against the real
# 26,887-transcript train split. The V100 took 8.1 min for the whole run (job 8784272).
#
# The point is not the hardware, it is the QUEUE. `gpu` was 24 running / 130 pending with a
# ~90 minute wait; `short` was 357 running with a 2-day limit and the HDF5 build started
# there within minutes. If training does not need a GPU then queueing it behind 130 GPU jobs
# is pure latency for nothing.
#
# WRITES TO A SEPARATE DIRECTORY, ON PURPOSE. It uses the SAME seed as GPU array task 1
# (100), so the two are directly comparable -- but two jobs writing
# best_model_{tag}_seed100.pt would be the exact collision utils.member_tag was added to
# prevent. --results-dir keeps the outputs apart; the HDF5 still comes from
# config_dn.yaml's data.hdf5_path, so both runs read the SAME input.
#
# Do not expect bit-identical results to the GPU member. verify_determinism.py is explicit
# that determinism is per (hardware, library version); AUC/AUPRC agreeing to ~3 decimals is
# the useful signal here, not equality.
#
# 03_train.py needs no change to run here: device is `cuda if torch.cuda.is_available() else
# cpu` (:117) and AMP is gated on device.type == "cuda" (:147), so both fall back correctly.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
# PY resolves in three steps so it works off this machine without changing behaviour on it:
# the authoring env if present, else whatever python3 is on PATH, else a loud failure. It
# previously defaulted to the authoring path unconditionally, which resolves for one account
# and silently points everyone else at a path that does not exist.
PY="${PY:-/home/p.castaldi/.conda/envs/nmd_model/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3 || true)"
[ -x "$PY" ] || { echo "FATAL: no python found. Set PY to one with torch and shap installed "\
                       "(see environment-model.yml)." >&2; exit 1; }

# OVERRIDABLE, AND DELIBERATELY *NOT* THE DEPOSIT-NATIVE TREE. This was a bare
# `OUT=results_4ct_dn_cpu` with no override, which the 2026-08-13 audit flagged with the other
# hardcoded trees. The fix is NOT to repoint it at results_deposit_h5_2026-08-04: this script
# TRAINS, so its output landing in the deposit-native tree would let a CPU smoke-test overwrite
# best_model_atg1000_stop1000_seed42.pt -- the deposited checkpoint every section 5 number rests
# on. A separate tree is correct here; the defect was that it could not be moved.
#
# The name follows the deprecated results_4ct_dn family and is kept only so existing logs still
# resolve. #SBATCH --output above names the same directory and CANNOT read this variable, because
# SBATCH directives are parsed before the shell runs -- so if you override OUT, the job log still
# lands in results_4ct_dn_cpu/.
OUT="${OUT:-results_4ct_dn_cpu}"
mkdir -p "$OUT"

SEED=$((SLURM_ARRAY_TASK_ID * 100))

# Let torch use the cores we asked Slurm for; the default can be the whole node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

TAG=$($PY paths_config.py --selected-tag --config config_dn.yaml)
if [ -z "$TAG" ]; then echo "FATAL: could not resolve selected tag" >&2; exit 1; fi
ATG=$(echo "$TAG" | sed -E 's/^atg([0-9]+)_stop([0-9]+)$/\1/')
STOP=$(echo "$TAG" | sed -E 's/^atg([0-9]+)_stop([0-9]+)$/\2/')

echo "=== node $(hostname) | CPU-only | seed=${SEED} | ${TAG} | threads=${OMP_NUM_THREADS} ==="
$PY -c "import torch; print('  torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '| threads:', torch.get_num_threads())"

# PREFLIGHT, seconds, before ~10 minutes of training. A doomed job should die now, not later.
$PY - <<'PRE'
import sys, yaml, h5py
cfg = yaml.safe_load(open("config_dn.yaml"))
h5 = cfg["data"]["hdf5_path"]
with h5py.File(h5, "r") as f:
    splits = {s.decode() if isinstance(s, bytes) else s for s in f["split"][:]}
need = {"train", "val"}
missing = need - splits
if missing:
    sys.exit(f"PREFLIGHT FAIL: {h5} lacks split(s) {sorted(missing)}; has {sorted(splits)}")
print(f"  preflight ok: {h5} has splits {sorted(splits)}")
PRE

echo "=== TRAIN (CPU) seed ${SEED} ==="
/usr/bin/time -v $PY 03_train.py --config config_dn.yaml --results-dir "$OUT" \
    --atg-window "${ATG}" --stop-window "${STOP}" --seed "${SEED}" 2>&1 | \
    grep -vE "^\s+(Voluntary|Involuntary|Swaps|File system|Socket|Signals|Page size|Average)" || true
echo "=== train exit: ${PIPESTATUS[0]} ==="

echo "=== EVALUATE (CPU) on val ==="
$PY evaluate.py --config config_dn.yaml --results-dir "$OUT" \
    --atg-window "${ATG}" --stop-window "${STOP}" --member-seed "${SEED}" --split val
echo "=== evaluate exit: $? ==="

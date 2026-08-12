#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ism_size
#SBATCH --output=results_ism_v6/size_launch_%j.log

# How long does the real bank actually take, per transcript, with the startup
# amortised? The only end-to-end number so far is a 3-transcript build where
# CUDA and cuDNN warmup was 50 of 53 seconds, and sizing a 5,000-transcript
# launch on that would be sizing it on the warmup.
#
# 50 transcripts from the HEAD OF THE SUBSET ORDER, which is the population the
# launch will actually build -- not the geometry extremes the correctness check
# used, because those were chosen to be unrepresentative on purpose.
#
# Three arms, one process each:
#   A  cached path, chunk 4,096   -- the current default
#   B  cached path, chunk 16,384  -- the profile gained 17.8x -> 21.4x with chunk
#                                    size, and peak memory was 1.22 GB of 32
#   C  the pre-cache builder      -- the honest end-to-end ratio at a size where
#                                    warmup no longer dominates either arm
#
# Each arm prints its own total and rate, and prints transcript 1 separately, so
# the steady-state rate can be taken as (total - first) rather than as a total
# that still carries the warmup.

cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
SC=results_ism_v6/size_check
mkdir -p $SC

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null
echo "=== code actually running (sha256, first 16) ==="
sha256sum build_ism_bank.py window_cache.py tensor_io.py | awk '{print "  " substr($1,1,16), $2}'
echo ""

N=50
CK=runs/interp_c32_b8_s100/best.pt

run_arm () {                    # $1 label   $2 script   $3 chunk   $4 out
  echo "=== ARM $1 ==="
  local t0=$SECONDS
  $PY "$2" --tensor results_tensor_v6 --checkpoint $CK \
      --split results_ism_v6/ism_subset.tsv \
      --n $N --chunk-rows "$3" --out "$4" --device cuda 2>&1 \
    | grep -E "^  \[|substitutions/s|peak GPU|valid positions|^  n "
  echo "  arm $1 wall: $((SECONDS - t0))s"
  echo ""
}

run_arm "A cached chunk 4096"  build_ism_bank.py  4096  $SC/size_a.h5
run_arm "B cached chunk 16384" build_ism_bank.py 16384  $SC/size_b.h5

# The pre-cache builder must run from the repo root: build_ism_bank resolves the
# pool table and the GENCODE flags from Path(__file__).parent, so from $SC it
# would look for $SC/results_pool_v6/orf_pool.tsv and die -- or worse, miss the
# GENCODE flags and build a bank with different columns.
REF=$SC/build_ism_bank_ref.py
if [ ! -f "$REF" ]; then echo "FATAL: $REF missing" >&2; exit 1; fi
echo "  reference builder sha256: $(sha256sum $REF | cut -d' ' -f1)"
cp $REF ./build_ism_bank_ref_run.py
run_arm "C pre-cache chunk 4096" ./build_ism_bank_ref_run.py 4096 $SC/size_c.h5
rm -f ./build_ism_bank_ref_run.py

echo "=== the three banks must still agree, or the rates describe different work ==="
$PY compare_banks.py $SC/size_a.h5 $SC/size_b.h5
echo "  A vs B exit: $?"
$PY compare_banks.py $SC/size_a.h5 $SC/size_c.h5
echo "  A vs C exit: $?"

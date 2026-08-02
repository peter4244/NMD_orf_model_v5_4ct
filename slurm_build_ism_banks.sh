#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ism_bank
#SBATCH --array=1-5
#SBATCH --output=results_ism_v6/bank_interp_%A_%a.log

# §9 step 8: one bank per seed of the interpretable arm, over the same 4,999
# transcript subset and the same discovery/confirmation split. Seeds differ in
# initialisation only, so a sequence feature present in one and absent in the
# other four is a property of that initialisation rather than of the model, and
# nothing in the gene-level split can detect that -- it holds the seed fixed and
# varies the transcripts. Requiring a finding across seeds varies the other axis.
#
# NOT the permuted-bin control: its discovery null is a circular shift on the real
# bank's own tracks and costs no forward passes, so banking it would spend half the
# budget answering a question nobody asked. NOT the nosqanti predictor: that arm is
# performance only.
#
# SIZING, measured (job 8885525, 50 transcripts from the head of the subset order):
#   chunk  4,096   4,315 substitutions/s   <- USED HERE
#   chunk 16,384   4,654 substitutions/s   (+8%, and NOT taken -- see below)
#   pre-cache        514 substitutions/s   (so the cache is 8.4x end to end)
# at ~7,500 substitutions per transcript that is ~2.4 h for 5,000, inside the
# 8-hour partition limit. The earlier 856/s was a 3-transcript build where CUDA
# warmup was 50 of 53 seconds, and is not the rate.
#
# WHY NOT 16,384, WHICH IS 8% FASTER. The same 50 transcripts built at 4,096 and
# at 16,384 are NOT the same bank: vals differ by up to 1.77e-06, vals_capture by
# 4.32e-07, and the reported batch_shape_offset itself moves 8.79e-07 -> 8.40e-07.
# That is not the cache -- cached and pre-cache builds at 4,096 are bitwise
# identical across all 37 datasets -- it is chunk shape changing float
# accumulation order, which the same-chunk baseline cancels WITHIN a chunk and not
# BETWEEN chunk choices.
#
# It matters because of where the capture arm lives: |vals_capture| has a median
# near 9e-05 with a large share of entries below 3e-06, so a 4e-07 wobble is a real
# fraction of the signal §5's mechanism claim rests on. Every verification, and
# every floor characterised so far, was at 4,096. Production runs at the shape that
# was verified; 8% does not buy an unverified shape.
#
# A RERUN MUST NOT MIX SHAPES. The builder skips shards already on disk, so shards
# from a run at a different chunk_rows have to be deleted, not reused -- otherwise
# the bank's floor varies by build order. The shard now records chunk_rows so a mix
# is visible after the fact; deleting is what prevents it.
#
# RESTARTABLE. One shard per transcript, written to a temporary name and renamed,
# so a shard is never half-written and a rerun skips what is on disk. A preemption
# costs one transcript, not a run.

cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python

SEED=$((SLURM_ARRAY_TASK_ID * 100))
CK=runs/interp_c32_b8_s${SEED}/best.pt
OUT=results_ism_v6/bank_interp_s${SEED}.h5

echo "=== node: $(hostname)   seed: ${SEED} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "=== code actually running (sha256, first 16) ==="
sha256sum build_ism_bank.py window_cache.py tensor_io.py | awk '{print "  " substr($1,1,16), $2}'
if [ ! -f "$CK" ]; then echo "FATAL: $CK missing" >&2; exit 1; fi
echo ""

$PY build_ism_bank.py --tensor results_tensor_v6 \
    --checkpoint "$CK" \
    --split results_ism_v6/ism_subset.tsv \
    --n 5000 --chunk-rows 4096 \
    --out "$OUT" --device cuda
rc=$?
echo "=== seed ${SEED} exit: $rc ==="
exit $rc

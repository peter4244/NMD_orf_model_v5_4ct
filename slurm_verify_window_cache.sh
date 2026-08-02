#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=wincache
#SBATCH --output=results_ism_v6/verify_window_cache_%j.log

# The three steps of the bank's verification bar, for the decode-cache change, in
# one job and in order. Every number in every bank is computed from the window the
# cache now builds, so this runs before any bank is built and the whole job stops
# at the first failure.
#
# It runs on the GPU because that is where the bank runs. The patched path builds
# the window on the device and the reference path builds it in numpy on the host,
# so device float behaviour is part of what step 1 is testing and a laptop pass
# does not answer for a V100. Laptop timings on this work also vary twofold on
# identical input, which is why the profile is here rather than local.
#
# No `set -e` around the checks themselves: a failure is a result to read. But a
# later step must not run on an earlier step's failure, so each gates the next.

cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
SC=results_ism_v6/cache_check
mkdir -p $SC

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null
# NOT `git rev-parse HEAD`. The scripts are copied onto this clone rather than
# pulled into it, so the clone's HEAD is the provenance of whatever was last
# committed HERE and says nothing about the code about to run. It read eadb797 on
# the first run of this job while the code was from 7ac672f on the laptop. The
# hashes below are the code that actually runs.
echo "=== code actually running (sha256, first 16) ==="
sha256sum tensor_io.py window_cache.py verify_window_cache.py compare_banks.py \
          build_ism_bank.py analysis_plans/profile_ism_cluster.py 2>/dev/null \
  | awk '{print "  " substr($1,1,16), $2}'
echo "=== this clone's HEAD, for reference only: $(git rev-parse --short HEAD) ==="
echo ""

echo "=== STEP 1: patched windows against decode_windows, on the GPU ==="
$PY verify_window_cache.py --tensor results_tensor_v6 --device cuda --n-cand 4000
rc=$?
echo "=== step 1 exit: $rc ==="
if [ $rc -ne 0 ]; then echo "STOP: the patched path is not the decoded path"; exit $rc; fi
echo ""

echo "=== PROFILE: where the time goes now, both paths, same rows, 5 repeats ==="
$PY analysis_plans/profile_ism_cluster.py
echo "=== profile exit: $? ==="
echo ""

echo "=== STEP 3a: a small bank, cached path ==="
# --only, not --n: the head of the order is whatever the order gives, and these
# three are chosen for geometry -- the two with the MOST candidates and the one
# with the FEWEST -- so the comparison covers the extremes rather than the typical.
# The fewest matters on its own: at K=1 the base pass and a chunk sit in different
# batch-shape regimes, which is the condition step 6's same-chunk baseline exists
# to cancel.
ONLY=$($PY - <<'EOF'
import h5py, numpy as np
f = h5py.File("results_tensor_v6/nmd_tensor.h5", "r")
iso, cnt = f["isoform_id"][:], f["count"][:]
import pandas as pd
sub = set(pd.read_csv("results_ism_v6/ism_subset.tsv", sep="\t")["isoform_id"])
keep = np.array([i for i, x in enumerate(iso) if x.decode() in sub])
lo = keep[np.argsort(cnt[keep])][:1]
hi = keep[np.argsort(cnt[keep])][-2:]
print(",".join(iso[i].decode() for i in np.concatenate([hi, lo])))
EOF
)
echo "  transcripts: $ONLY"
# An empty $ONLY is not an empty job: build_ism_bank falls back to --n, whose
# default is 1,000 transcripts. The selection failing would then look like the
# verification running, until the queue killed it at the wall clock.
if [ -z "$ONLY" ]; then echo "FATAL: transcript selection produced nothing" >&2; exit 1; fi
$PY build_ism_bank.py --tensor results_tensor_v6 \
    --checkpoint runs/interp_c32_b8_s100/best.pt \
    --split results_ism_v6/ism_subset.tsv \
    --only "$ONLY" --out $SC/bank_cached.h5 --device cuda
rc=$?
echo "=== step 3a exit: $rc ==="
if [ $rc -ne 0 ]; then echo "STOP: the cached build failed"; exit $rc; fi
echo ""

echo "=== STEP 3b: the same bank, pre-change decode path ==="
# The reference builder is commit 0199a82's build_ism_bank.py -- the state just
# before the cache. It is COPIED UP rather than taken from git here, because this
# clone is not guaranteed to hold that commit and `git show` would then fail late,
# after step 1 and the profile had already spent their time. It is not kept in the
# tree either: a frozen copy of a superseded script drifts, with nothing to say when.
#
# Its sha256 goes in the log, so which reference produced the comparison is a fact
# in the artifact rather than an assumption about what was copied.
REF=$SC/build_ism_bank_ref.py
if [ ! -f "$REF" ]; then
  echo "FATAL: $REF is missing. Generate it locally with" >&2
  echo "  git show 0199a82:build_ism_bank.py > build_ism_bank_ref.py" >&2
  echo "and copy it to $REF before submitting." >&2
  exit 1
fi
echo "  reference builder sha256: $(sha256sum $REF | cut -d' ' -f1)"
PYTHONPATH=$PWD $PY $REF --tensor results_tensor_v6 \
    --checkpoint runs/interp_c32_b8_s100/best.pt \
    --split results_ism_v6/ism_subset.tsv \
    --only "$ONLY" --out $SC/bank_decoded.h5 --device cuda
rc=$?
echo "=== step 3b exit: $rc ==="
if [ $rc -ne 0 ]; then echo "STOP: the reference build failed"; exit $rc; fi
echo ""

echo "=== STEP 3c: the two banks, entry by entry ==="
$PY compare_banks.py $SC/bank_cached.h5 $SC/bank_decoded.h5
rc=$?
echo "=== step 3c exit: $rc ==="
if [ $rc -ne 0 ]; then echo "STOP: the cache changed the bank"; exit $rc; fi
echo ""

echo "=== STEP 2: the bank against the model's own forward(), no shared path ==="
$PY verify_ism_bank.py --tensor results_tensor_v6 \
    --checkpoint runs/interp_c32_b8_s100/best.pt \
    --bank $SC/bank_cached.h5 --transcripts 3 --device cuda
echo "=== step 2 exit: $? ==="

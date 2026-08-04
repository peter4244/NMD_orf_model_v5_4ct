#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=00:30:00   # measured 9:18; ~3-4x headroom. Right-sized 2026-07-29:
#   an oversized request is excluded from Slurm backfill, which is what kept job
#   8826175 at PENDING(Priority) for an hour with 14 suitable nodes idle.
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=dn_h5
#SBATCH --output=results_deposit_h5_2026-08-04/build_h5_%j.log

# DEPOSIT-NATIVE HDF5 build (2026-07-27).
#
# Differs from slurm_build_h5.sh in two ways, both deliberate:
#   1. --results-dir results_4ct_dn, so the published results_4ct and the old 6-CT tree
#      it symlinks into are untouched. Every input here was regenerated from the deposit.
#   2. the conda env python is called by ABSOLUTE PATH rather than `conda activate`.
#      Conda init is being OOM-killed on the login node (.bashrc:33); an absolute path
#      cannot be killed by a shell-init that never runs.
set -euo pipefail
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python

echo "=== Building DEPOSIT-NATIVE HDF5 ==="
$PY -V
RESULTS_DIR="${RESULTS_DIR:-results_deposit_h5_2026-08-04}"
# ONE VARIABLE (2026-08-04). The build wrote to one directory and the verify block below
# read a DIFFERENT hardcoded one, so job 8934670 built a complete, correct HDF5 and then
# exited 1 on a stale path -- reporting FAILED for a run that succeeded. A job that
# misreports its outcome is the same defect class as one that exits 0 on a broken tree,
# and it is worse here because the honest signal is the only thing distinguishing this
# rebuild from the one it replaces.
$PY data_prep.py --config config_dn.yaml --results-dir "$RESULTS_DIR" --workers 8

echo "=== Verifying ==="
$PY -c "
import h5py, json, collections
with h5py.File(\"$RESULTS_DIR/nmd_orf_data.h5\",\"r\") as f:
    print(\"Keys:\", list(f.keys()))
    print(\"orf_features:\", f[\"orf_features\"].shape)
    print(\"n isoforms:\", f[\"orf_mask\"].shape[0])
    sp = [s.decode() if isinstance(s,bytes) else s for s in f[\"split\"][:]]
    print(\"splits:\", collections.Counter(sp))
    # \"labels\", not \"label\" -- the dataset data_prep.py writes is plural (2026-07-29).
    # This one-character typo is why job 8783842 shows FAILED in sacct: data_prep.py had
    # already written a complete 4.18 GB HDF5, and only this post-build verification block
    # raised KeyError. Under `set -euo pipefail` that marked a successful build as a failed
    # job, which is worse than no check at all -- a verification step that fails on its own
    # bug teaches you to distrust the verification.
    lab = f[\"labels\"][:]
    print(\"labels: NMD\", int(lab.sum()), \"/ non-NMD\", int((lab==0).sum()))
    print(\"window_sizes:\", json.loads(f.attrs[\"window_sizes\"]))
"
echo "=== Done ==="

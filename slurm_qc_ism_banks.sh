#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=qc_banks
#SBATCH --output=results_ism_v6/qc_banks_%j.log

# Is the bank complete, and is any of it readable above its own floor? A finished
# job and a written file are not evidence of either. Memory is generous because
# the cross-seed comparison holds vals and vals_capture for every seed at once:
# (4999, 9833, 4) float32 is 786 MB per array per bank.

cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
echo "=== code (sha256, first 16) ==="
sha256sum qc_ism_banks.py | awk '{print "  " substr($1,1,16), $2}'
echo ""
$PY qc_ism_banks.py results_ism_v6/bank_interp_s*.h5
echo "=== qc exit: $? ==="

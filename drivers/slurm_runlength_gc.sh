#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=md_gcctl
#SBATCH --output=results_ism_v6/model_runlength_gc_%j.log

# THE DECISIVE CONTROL for the run-length result. Channel 5 averages GC over +/-25
# bases, so adjacent positions share 49 of the 51 positions in their GC window. If
# the effect is driven by the GC shift a substitution causes, adjacent positions
# would correlate BY CONSTRUCTION and the clustering would be an encoding artifact.
#
# Exactly one of the three substitutions at a position preserves GC status (A<->T,
# C<->G) -- verified in the bank, 3 finite entries per position of which 1 is
# GC-neutral. If clustering survives on that one, GC smoothing is not the driver.

cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
echo "=== code actually running (sha256, first 16) ==="
sha256sum analysis_runlength_replicate.py | awk '{print "  " substr($1,1,16), $2}'
echo ""
for SUB in all neutral changing; do
  echo "########## subset = $SUB ##########"
  $PY analysis_runlength_replicate.py results_ism_v6/bank_interp_s100.h5 \
      --subset $SUB --rule fraction --k 0.01 --k-sweep 0.005,0.01,0.02,0.05
  echo ""
done
echo "=== exit: $? ==="

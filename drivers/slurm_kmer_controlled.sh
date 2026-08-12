#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=03:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=md_kmerc
#SBATCH --output=results_ism_v6/model_kmer_controlled_%j.log

# The AU-enrichment claim with BOTH confounds closed in one run.
#   GC       rank positions using only the GC-preserving substitution
#   REGIONAL compute enrichment within each region against that region's own
#            positions, so foreground and background share a compositional pool
# The two are not substitutes: holding GC constant does not move where the elevated
# positions are. Job 8886938 was meant to answer the first and crashed before
# reaching any k-mer output, so BOTH are currently open.

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-/home/p.castaldi/.conda/envs/nmd_model/bin/python}"
echo "=== code actually running (sha256, first 16) ==="
sha256sum analysis_kmer_controlled.py | awk '{print "  " substr($1,1,16), $2}'
echo ""
for K in 5 6; do
  echo "################ k=$K ################"
  $PY analysis_kmer_controlled.py results_ism_v6/bank_interp_s100.h5 --k $K
  echo ""
done
echo "=== exit: $? ==="

#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=03:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=md_anchor
#SBATCH --output=results_ism_v6/model_kmer_anchors_%j.log

# The region boundary both ways. "Downstream of the stop" needs an ORF, and the two
# candidates disagree for most transcripts:
#   selected   the max-p_select candidate -- the ORF whose decay is being scored
#   reference  the annotated candidate where one exists (3,422 of 4,999)
# Neither is obviously right. If the AU result depends on which, that dependence is
# the finding, and it is cheaper to learn it now than to defend a choice later.
#
# GC-neutral scoring is NOT run: measured today, that restriction drives GC at
# elevated positions from 0.501 to 0.682 and its k-mer output is an artifact of the
# control. The elevated set is not GC-biased under normal scoring (0.503 vs 0.501),
# so the confound it was built for does not apply.

cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
echo "=== code actually running (sha256, first 16) ==="
sha256sum analysis_kmer_controlled.py | awk '{print "  " substr($1,1,16), $2}'
echo ""
for ANCHOR in selected reference; do
  echo "################ region anchor = $ANCHOR ################"
  $PY analysis_kmer_controlled.py results_ism_v6/bank_interp_s100.h5 \
      --k 5 --region-anchor $ANCHOR
  echo ""
done
echo "=== exit: $? ==="

#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=dn_exports
#SBATCH --output=results_deposit_h5_2026-08-04/export_chain_%j.log
#
# The section 5 export chain on the post-clip model, full cohort (D74/D77).
#
# WHY A COMPUTE NODE AND 64G. 08 and 09b load the whole five-replicate npz set at once and die
# rc=137 on a login node against a per-user cap -- not node memory, which is why the number here is
# generous rather than measured to the edge.
#
# WHY --tag CARRIES THE SEED. utils.member_tag composes a member stem as {tag}_seed{N}, so passing
# the composed stem as --tag makes every deepshap_*_{tag}_run{N}.npz lookup resolve to the files
# deepshap.py actually wrote. That is the convention, not a trick. 08 is the exception: it builds
# its own tag from --atg/--stop and has no --tag, so it takes --member-seed directly.
#
# WHY --split all APPEARS ONLY ON SOME. Only the scripts that read a predictions file need it,
# because evaluate.py puts the split in that filename. The array-only scripts do not have the flag
# and do not need it.
#
# NO `set -e`, DELIBERATELY. Each step's exit code is captured and reported, and the job exits
# non-zero if any failed. One broken step should not hide the status of the nine after it -- the
# whole point of running these together is to learn which work in a single pass.
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
RESULTS_DIR="${RESULTS_DIR:-results_deposit_h5_2026-08-04}"
# Windows and tag come from the config, never from literals here -- 39 such literals
# across 26 drivers would otherwise keep reading 500/500 after a re-selection.
TAG=$($PY paths_config.py --selected-tag --config config_dn.yaml) || exit 1
ATG=${TAG#atg}; ATG=${ATG%%_*}; STOP=${TAG##*stop}
cd "$SLURM_SUBMIT_DIR" || exit 1
RES=${RESULTS_DIR:-results_deposit_h5_2026-08-04}
CFG=config_dn.yaml
MTAG=${TAG}_seed42

echo "=== node $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true
echo "results-dir=$RES  member-tag=$MTAG  split=all"
echo

overall=0
step () {                       # step <label> <command...>
  local label="$1"; shift
  echo "--- $label ---"
  "$@"
  local rc=$?
  echo "--- $label exit: $rc ---"
  echo
  [ $rc -ne 0 ] && overall=$rc
  return 0
}

# 06 needs --run-id EXPLICITLY: its default is None, which means "the un-suffixed npz", and no
# such file exists once replicates are written as _run{N}. 07 defaults to 1 and is passed it
# anyway so the two are visibly on the same replicate rather than coincidentally so.
step "06 deepshap tsv"        $PY 06_export_deepshap_tsv.py --results-dir "$RES" --tag "$MTAG" --run-id 1
step "07 motif analysis"      $PY 07_motif_analysis.py --results-dir "$RES" --tag "$MTAG" --run-id 1
step "08 subgroup deepshap"   $PY 08_export_subgroup_deepshap_tsv.py --results-dir "$RES" \
                                  --atg "$ATG" --stop "$STOP" --n-runs 5 --member-seed 42 --split all
step "09 gc content"          $PY 09_export_gc_content.py --config "$CFG" --results-dir "$RES" --tag "$MTAG"
step "09 junction ordinal"    $PY 09_export_junction_ordinal.py --config "$CFG" --results-dir "$RES" --tag "$MTAG"
step "09 polya"               $PY 09_export_polya.py --config "$CFG" --results-dir "$RES" --tag "$MTAG" --split all
step "09b subgroup profiles"  $PY 09b_export_subgroup_profiles.py --config "$CFG" --results-dir "$RES" \
                                  --tag "$MTAG" --n-runs 5 --split all
step "09c signed gc channel"  $PY 09c_export_signed_gc_channel.py --config "$CFG" --results-dir "$RES" \
                                  --tag "$MTAG" --n-runs 5
step "09d gc refaug only"     $PY 09d_export_gc_content_refaug_only.py --config "$CFG" --results-dir "$RES" --tag "$MTAG"

echo "=== export chain overall exit: $overall ==="
exit $overall

#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=render_dn
#SBATCH --output=results_deposit_h5_2026-08-04/render_dn_%j.log

# DEPOSIT-NATIVE port of drivers/slurm_render_v5.sh, 2026-08-12.
# The original reproduces the superseded sweep-era run: it targets results_4ct and
# config.yaml, whose selection was atg500_stop500. This one targets the deposit-native
# tree and reads the current selection from config_dn.yaml.

set -o pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
# PY resolves in three steps so it works off this machine without changing behaviour on it:
# the authoring env if present, else whatever python3 is on PATH, else a loud failure. It
# previously defaulted to the authoring path unconditionally, which resolves for one account
# and silently points everyone else at a path that does not exist.
PY="${PY:-/home/p.castaldi/.conda/envs/nmd_model/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3 || true)"
[ -x "$PY" ] || { echo "FATAL: no python found. Set PY to one with torch and shap installed "\
                       "(see environment-model.yml)." >&2; exit 1; }
RESULTS_DIR="${RESULTS_DIR:-results_deposit_h5_2026-08-04}"
# THE SWEEP IS A SECOND TREE. Section 1 describes the window sweep, every other section describes
# the ONE selected model, and they live in different directories -- results_deposit_h5_2026-08-04
# trains a single configuration and holds none of the twelve. Passed explicitly rather than left to
# the report's default, so the dependency is visible in the job log.
SWEEP_DIR="${SWEEP_DIR:-results_sweep_dn_2026-08-04}"
# The window and tag come from the config, never from literals here.
TAG=$($PY paths_config.py --selected-tag --config config_dn.yaml) || { echo "cannot read selected tag" >&2; exit 1; }
ATG=${TAG#atg}; ATG=${ATG%%_*}; STOP=${TAG##*stop}
echo "=== node $(hostname) | tag=$TAG (atg=$ATG stop=$STOP) | results=$RESULTS_DIR | sweep=$SWEEP_DIR ==="
[ -d "$SWEEP_DIR" ] || echo "WARNING: sweep dir $SWEEP_DIR not found -- Section 1 will fail its guard"

# THIS RENDERS INSIDE THE CONTAINER, AND THE BARE-Rscript VERSION NEVER WORKED. Until 2026-08-15
# this line was a plain `Rscript -e rmarkdown::render(...)`. Rscript is NOT on the compute-node
# PATH on Explorer -- job 9194174 died `Rscript: command not found`, exit 127 -- and the node's
# module R would not have helped: the section 5 report needs pROC and ggseqlogo, which nmd_1.2.sif
# lacks and nmd_1.3.sif carries. The first successful end-to-end render of this report was job
# 9209766, inside nmd_1.3.sif.
SIF="${SIF:-/scratch/p.castaldi/clean_room_2026.8.13/nmd_1.3.sif}"
DEPOSIT_ROOT="${DEPOSIT_ROOT:-$HOME/cc/nmd_deposit_2026}"
OUT_DIR="${OUT_DIR:-$PWD/render_out}"
TMP_DIR="${TMP_DIR:-/scratch/$USER/render_tmp_${SLURM_JOB_ID:-$$}}"
mkdir -p "$OUT_DIR" "$TMP_DIR"

[ -r "$SIF" ] || { echo "FATAL: no container at $SIF. Set SIF." >&2; exit 1; }

# TMPDIR IS REDIRECTED because --containall gives a 64 MB tmpfs /tmp with TMPDIR unset, and
# RSQLite then reports "database or disk is full" with terabytes free on /scratch (W343).
#
# DEPOSIT POINTS AT source_data/model, NOT source_data. That is make_architecture_figure.R's
# reading of $DEPOSIT -- it globs "$DEPOSIT/metrics_atg*_stop*_test_clean.json" directly -- while
# config/paths.yml and data_export_deposit.py define DEPOSIT as source_data. The figure script's
# own comment claims it reads "the same variable"; it does not. Set for what the script wants and
# recorded here, because a reader following paths.yml gets "found 0" and a refusal to draw.
apptainer exec --containall \
  --bind "$PWD":"$PWD" --bind "$DEPOSIT_ROOT":"$DEPOSIT_ROOT" \
  --bind "$OUT_DIR":"$OUT_DIR" --bind "$TMP_DIR":"$TMP_DIR" \
  --pwd "$PWD" \
  --env NMD_RESULTS_DIR="$RESULTS_DIR" \
  --env NMD_SWEEP_DIR="$SWEEP_DIR" \
  --env DEPOSIT="$DEPOSIT_ROOT/source_data/model" \
  --env TMPDIR="$TMP_DIR" --env SQLITE_TMPDIR="$TMP_DIR" \
  "$SIF" \
  Rscript -e "rmarkdown::render('orf_model_report_v5.Rmd', knit_root_dir=normalizePath('.'), output_dir='$OUT_DIR')"
rc=$?
echo "=== render exit: $rc ==="
ls -l "$OUT_DIR" 2>/dev/null
rm -rf "$TMP_DIR"
# PROPAGATE THE RENDER STATUS, NOT rm's. A first version of this ended on the cleanup line and
# SLURM reported COMPLETED 0:0 while the render had exited 1 -- the same trap the DeepSHAP
# wrappers document, reproduced while fixing something else.
exit $rc

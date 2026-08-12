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
PY="${PY:-/home/p.castaldi/.conda/envs/nmd_model/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-results_deposit_h5_2026-08-04}"
# The window and tag come from the config, never from literals here.
TAG=$($PY paths_config.py --selected-tag --config config_dn.yaml) || { echo "cannot read selected tag" >&2; exit 1; }
ATG=${TAG#atg}; ATG=${ATG%%_*}; STOP=${TAG##*stop}
echo "=== node $(hostname) | tag=$TAG (atg=$ATG stop=$STOP) | results=$RESULTS_DIR ==="

# rmarkdown::render() does not pass arguments through; NMD_RESULTS_DIR is the override.
NMD_RESULTS_DIR="$RESULTS_DIR" Rscript -e 'rmarkdown::render("orf_model_report_v5.Rmd", knit_root_dir = normalizePath("."))'
rc=$?; echo "=== render exit: $rc ==="; exit $rc

echo "=== done ==="

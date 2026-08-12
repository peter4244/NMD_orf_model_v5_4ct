#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=v5_rpt
#SBATCH --output=results_4ct/render_v5_%j.log

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
eval "$(conda shell.bash hook)"
conda activate nmd_model

echo "=== Rendering orf_model_report_v5.Rmd ==="
Rscript -e 'rmarkdown::render("orf_model_report_v5.Rmd", output_dir = ".")'

echo "=== Done ==="

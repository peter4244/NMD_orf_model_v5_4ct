#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=v5_rpt
#SBATCH --output=results/render_v5_%j.log

cd /home/p.castaldi/cc/nmd_orf_model_v5
eval "$(conda shell.bash hook)"
conda activate nmd_model

echo "=== Rendering orf_model_report_v5.Rmd ==="
Rscript -e 'rmarkdown::render("orf_model_report_v5.Rmd", output_dir = ".")'

echo "=== Done ==="

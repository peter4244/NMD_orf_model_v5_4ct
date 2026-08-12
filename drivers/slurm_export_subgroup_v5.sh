#!/bin/bash
#SBATCH --job-name=export_subgroup
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=results_4ct/export_subgroup_%j.log

source activate nmd_model
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

python 08_export_subgroup_deepshap_tsv.py --atg 500 --stop 500 --n-runs 5

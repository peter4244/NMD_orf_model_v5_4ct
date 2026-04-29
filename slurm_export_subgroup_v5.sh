#!/bin/bash
#SBATCH --job-name=export_subgroup
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/export_subgroup_%j.log

source activate nmd_model
cd /home/p.castaldi/cc/nmd_orf_model_v5

python 08_export_subgroup_deepshap_tsv.py --atg 500 --stop 500 --n-runs 5

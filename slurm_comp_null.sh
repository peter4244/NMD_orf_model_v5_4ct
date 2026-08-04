#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=md_compnull
#SBATCH --output=results_ism_v6/model_comp_null_%j.log
cd /home/p.castaldi/cc/nmd_orf_model_v5_4ct
/home/p.castaldi/.conda/envs/nmd_model/bin/python comp_null.py

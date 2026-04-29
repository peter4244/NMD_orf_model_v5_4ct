#!/bin/bash
# Submit a single ORF DeepSHAP job on Explorer
# Usage: bash submit_orf.sh <orf_index>
# Example: bash submit_orf.sh 2

ORF=$1
if [ -z "$ORF" ]; then
    echo "Usage: bash submit_orf.sh <orf_index>"
    exit 1
fi

cd /home/p.castaldi/cc/nmd_orf_model_v5
sbatch --partition=gpu --gres=gpu:1 --time=08:00:00 --mem=32G --cpus-per-task=4 --job-name=v5_orf${ORF} --output=results/deepshap_joint_orf${ORF}_%j.log --wrap="cd /home/p.castaldi/cc/nmd_orf_model_v5 && eval \"\$(conda shell.bash hook)\" && conda activate nmd_model && python deepshap.py --config config.yaml --n-explain 0 --n-background 500 --atg-window 500 --stop-window 500 --seed 100 --run-id 1 --branches joint --orf-index ${ORF}"

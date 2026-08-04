#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=dn_inferall_fix
#SBATCH --output=results_4ct_dn/inferall_dn_fixed_%j.log

# REDO of the full-cohort inference. The first attempt (8817545) produced a HYBRID and looked fine.
#
# WHAT WENT WRONG. run_infer_all.py takes --results-dir, and that correctly selected the
# deposit-native CHECKPOINT. But the universe comes from a different place entirely:
#     h5_path = config["data"]["hdf5_path"]        # run_infer_all.py:77
# and --config defaults to config.yaml, whose hdf5_path is results_4ct/nmd_orf_data.h5 -- the
# PUBLISHED universe. So it scored the deposit-native model over 39,938 published isoforms
# (splits 25441/10131/4236/130) and exited 0. config_dn.yaml exists for precisely this and was
# not passed.
#
# THE LESSON FOR THE GUARD. The previous run's guard compared the CHECKPOINT against the
# published one and passed, because the checkpoint was right. A guard has to check the thing
# that can be wrong, not the thing you happened to think of -- so this one asserts the OUTPUT
# ROW COUNT is the deposit-native universe, which is the property that actually failed.
set -euo pipefail
PY=/home/p.castaldi/.conda/envs/nmd_model/bin/python
cd "$SLURM_SUBMIT_DIR"
EXPECTED=42043

echo "=== node $(hostname) ==="
echo "config_dn.yaml hdf5_path: $(grep hdf5_path config_dn.yaml)"

$PY run_infer_all.py --config config_dn.yaml --results-dir results_4ct_dn \
    --split all --atg-window 500 --stop-window 500

OUT=results_4ct_dn/predictions_all_atg500_stop500.tsv
N=$(( $(wc -l < "$OUT") - 1 ))
echo
echo "=== guard: output universe ==="
echo "  rows=$N  expected=$EXPECTED"
if [ "$N" -ne "$EXPECTED" ]; then
  echo "FATAL: wrong universe -- $N rows, expected $EXPECTED. This is the 8817545 failure."
  exit 1
fi
echo "  OK"
echo
head -1 "$OUT"
awk -F'\t' 'NR>1{c[$3]++} END{for(k in c) printf "  %-14s %d\n", k, c[k]}' "$OUT"
ls -la "$OUT"

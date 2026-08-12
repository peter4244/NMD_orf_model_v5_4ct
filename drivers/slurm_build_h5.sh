#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=4ct_h5
#SBATCH --output=results_4ct/build_h5_%j.log

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
eval "$(conda shell.bash hook)"
conda activate nmd_model

echo "=== Building 4ct HDF5 dataset ==="
python data_prep.py --results-dir results_4ct --workers 8

echo "=== Verifying ==="
python -c "
import h5py, json
with h5py.File('results_4ct/nmd_orf_data.h5','r') as f:
    print('Keys:', list(f.keys()))
    print('orf_features:', f['orf_features'].shape)
    print('orf_mask:', f['orf_mask'].shape)
    print('n_seq_channels:', f.attrs['n_seq_channels'])
    print('orf_feature_cols:', json.loads(f.attrs['orf_feature_cols']))
    print('window_sizes:', json.loads(f.attrs['window_sizes']))
    for ws in json.loads(f.attrs['window_sizes']):
        print(f'  w{ws}/atg: {f[\"w\" + str(ws)][\"atg_windows\"].shape}')
    print('Has tx_features:', 'tx_features' in f)
"

echo "=== Done ==="

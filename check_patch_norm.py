import numpy as np
from pathlib import Path

# Load a preprocessed patch to test
preprocess_dir = Path('data/runs/run_final_verify/processed')
all_patches = list(preprocess_dir.glob('*/patches/*.npz'))
print(f'Found {len(all_patches)} patches')

if all_patches:
    patch_file = all_patches[0]
    print(f'Loading test patch: {patch_file}')
    npz_file = np.load(str(patch_file))
    patch_data = npz_file['patch']
    print(f'Patch shape: {patch_data.shape}')
    print(f'Patch dtype: {patch_data.dtype}')
    print(f'Patch min: {patch_data.min():.6f}')
    print(f'Patch max: {patch_data.max():.6f}')
    print(f'Patch mean: {patch_data.mean():.6f}')
    print(f'Patch std: {patch_data.std():.6f}')
    
    # Show per-band stats
    print(f'\nPer-band statistics:')
    for i in range(patch_data.shape[0]):
        band = patch_data[i]
        print(f'  Band {i:2d}: min={band.min():.4f}, max={band.max():.4f}, mean={band.mean():.4f}, std={band.std():.4f}')
else:
    print('No patches found')

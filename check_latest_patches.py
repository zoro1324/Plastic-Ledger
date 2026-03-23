import numpy as np
from pathlib import Path

# Check the latest patches with the corrected clipping
patch_dirs = sorted([d for d in Path('data/runs').glob('run_*/processed/*/patches')], 
                    key=lambda p: p.parent.parent.stat().st_mtime, reverse=True)

for patch_dir in patch_dirs[:2]:
    print(f"\nChecking directory: {patch_dir.parent.parent.name}")
    patches = list(patch_dir.glob('patch_*.npz'))
    print(f"  Found {len(patches)} patches")
    
    if patches:
        # Load first patch
        npz = np.load(str(patches[0]))
        patch_key = list(npz.files)[0]
        patch_data = npz[patch_key].astype(np.float32)
        
        print(f"  First patch shape: {patch_data.shape}")
        print(f"  First patch range: min={patch_data.min():.4f}, max={patch_data.max():.4f}")
        print(f"  First patch mean: {patch_data.mean():.4f}, std: {patch_data.std():.4f}")
        
        # Per-band analysis
        has_variation = 0
        for i in range(patch_data.shape[0]):
            band = patch_data[i]
            if band.std() > 0.01:  # Has real variation
                has_variation += 1
                if has_variation <= 3:  # Show first 3 bands with variation
                    print(f"    Band {i}: min={band.min():.4f}, max={band.max():.4f}, std={band.std():.4f}")
        
        print(f"  Total bands with variation: {has_variation}/11")

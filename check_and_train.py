"""Quick status check and train-only runner"""
from pathlib import Path

# Check balanced dataset
balanced_patches = Path("d:/Plastic-Ledger/U-net-models/dataset/MARIDA_BALANCED/patches")
if balanced_patches.exists():
    tif_files = list(balanced_patches.rglob("*.tif"))
    image_files = [f for f in tif_files if "_cl" not in f.name]
    mask_files = [f for f in tif_files if "_cl" in f.name]
    print(f"✓ Balanced dataset created:")
    print(f"  Total TIF files: {len(tif_files)}")
    print(f"  Image files: {len(image_files)}")
    print(f"  Mask files: {len(mask_files)}")
else:
    print(f"✗ Balanced dataset not found")

# Check splits
splits_dir = balanced_patches.parent / "splits"
if splits_dir.exists():
    train_file = splits_dir / "train_X.txt"
    if train_file.exists():
        train_patches = train_file.read_text().strip().splitlines()
        print(f"  Train patches in split: {len(train_patches)}")
    
print("\nReady to train model...")

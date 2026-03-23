import os
import rasterio
import numpy as np

dataset_dir = r"d:\Plastic-Ledger\U-net-models\dataset\MARIDA"
patches_dir = os.path.join(dataset_dir, "patches")
splits_dir = os.path.join(dataset_dir, "splits")

def analyze_splits():
    print("--- SPLIT ANALYSIS ---")
    total_files = 0
    for split in ["train_X.txt", "val_X.txt", "test_X.txt"]:
        split_file = os.path.join(splits_dir, split)
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                print(f"{split}: {len(lines)} files")
                total_files += len(lines)
        else:
            print(f"{split}: NOT FOUND")
    print(f"Total listed patches: {total_files}")

def get_patch_files():
    subdirs = [os.path.join(patches_dir, d) for d in os.listdir(patches_dir) if os.path.isdir(os.path.join(patches_dir, d))]
    if not subdirs:
        return None, None
        
    test_dir = subdirs[0]
    tifs = [f for f in os.listdir(test_dir) if f.endswith(".tif") and not f.endswith("_cl.tif") and not f.endswith("_conf.tif")]
    
    if not tifs:
        return None, None
        
    img_path = os.path.join(test_dir, tifs[0])
    mask_path = os.path.join(test_dir, tifs[0].replace(".tif", "_cl.tif"))
    return img_path, mask_path

def analyze_patch():
    print("\n--- PATCH ANALYSIS ---")
    img_path, mask_path = get_patch_files()
    
    if not img_path:
        print("No patches found to analyze.")
        return

    print(f"Sample Image: {img_path}")
    print(f"Sample Mask:  {mask_path}")

    # Analyze Image
    with rasterio.open(img_path) as src:
        img = src.read()
        print(f"Image Shape (Bands, H, W): {img.shape}")
        print(f"Image Dtype: {img.dtype}")
        print(f"Image Min: {img.min()}, Max: {img.max()}")
        
    # Analyze Mask
    if os.path.exists(mask_path):
        with rasterio.open(mask_path) as src:
            mask = src.read()
            print(f"Mask Shape: {mask.shape}")
            print(f"Mask Dtype: {mask.dtype}")
            unique_classes = np.unique(mask)
            print(f"Unique classes in this mask: {unique_classes}")
    else:
        print("Mask file not found!")

if __name__ == "__main__":
    analyze_splits()
    analyze_patch()

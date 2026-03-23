import os
import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset

class MaridaDataset(Dataset):
    def __init__(self, splits_file, patches_dir):
        self.patches_dir = patches_dir
        with open(splits_file, 'r') as f:
            # MARIDA splits list "patch_name.tif"
            self.image_names = [line.strip() for line in f.readlines() if line.strip()]
            
        if len(self.image_names) > 0:
            first_img_path = self._get_image_path(self.image_names[0])
            with rasterio.open(first_img_path) as src:
                self.in_channels = src.count
        else:
            self.in_channels = 11 # fallback

    def _get_image_path(self, img_name):
        # E.g. "1-12-19_48MYU_0" -> folder "S2_1-12-19_48MYU", file "S2_1-12-19_48MYU_0.tif"
        real_img_name = f"S2_{img_name}.tif"
        parts = real_img_name.split('_')
        folder_name = "_".join(parts[:-1])
        return os.path.join(self.patches_dir, folder_name, real_img_name)

    def _get_mask_path(self, img_name):
        img_path = self._get_image_path(img_name)
        return img_path.replace(".tif", "_cl.tif")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = self._get_image_path(img_name)
        mask_path = self._get_mask_path(img_name)
        
        with rasterio.open(img_path) as src:
            image = src.read() # (C, H, W)
            # Sentinel-2 normalization (approx 0-1 range)
            image = image.astype(np.float32) / 10000.0
            # Protect against NoData/NaNs causing exploding gradients
            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
            image = np.clip(image, 0.0, 1.0)
            
        with rasterio.open(mask_path) as src:
            mask_raw = src.read(1).astype(np.int32) # (H, W)
            # The raw MARIDA dataset maps 0 to Background. Classes exist 1-15.
            # We map 0 -> 255 (so PyTorch organically ignores it) and map 1-15 -> 0-14
            mask = np.where(mask_raw > 0, mask_raw - 1, 255).astype(np.int64)
            
        # Segformer expects PyTorch tensors
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)
        
        return {"pixel_values": image, "labels": mask}

"""
Marine Debris Focused Evaluation
===================================
Evaluates ONLY on test patches that contain at least one Marine Debris pixel.
This gives a focused, honest view of the model's debris detection performance.
"""

import os, sys
sys.path.insert(0, r"d:\Plastic-Ledger\SegFormer-Model")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rasterio
from torch.utils.data import Dataset, DataLoader

# ── Config ───────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR    = r"d:\Plastic-Ledger\U-net-models\dataset\MARIDA"
PATCHES_DIR = os.path.join(DATA_DIR, "patches")
TEST_SPLIT  = os.path.join(DATA_DIR, "splits", "test_X.txt")
MODEL_PATH  = r"d:\Plastic-Ledger\best-models\best_model_SegTransformer.pth"
NUM_CLASSES = 15
IGNORE_IDX  = 255
DEBRIS_CLS  = 0   # class index 0 = Marine Debris (raw label 1 → 1-1=0)

# ── Dataset ──────────────────────────────────────────────────────────────────
class MaridaTestDataset(Dataset):
    def __init__(self, splits_file, patches_dir):
        self.patches_dir = patches_dir
        with open(splits_file) as f:
            self.all_names = [l.strip() for l in f if l.strip()]
        self.in_channels = self._detect_channels()
        # Pre-filter: keep only patches that contain Marine Debris (raw label = 1)
        self.names = self._filter_debris_patches()
        print(f"  Total test patches:          {len(self.all_names)}")
        print(f"  Patches WITH Marine Debris:  {len(self.names)}")

    def _detect_channels(self):
        with rasterio.open(self._img_path(self.all_names[0])) as s:
            return s.count

    def _filter_debris_patches(self):
        debris_patches = []
        for name in self.all_names:
            try:
                with rasterio.open(self._mask_path(name)) as s:
                    raw = s.read(1)
                    if (raw == 1).any():   # raw label 1 = Marine Debris
                        debris_patches.append(name)
            except Exception:
                pass
        return debris_patches

    def _img_path(self, name):
        real = f"S2_{name}.tif"
        parts = real.split("_")
        folder = "_".join(parts[:-1])
        return os.path.join(self.patches_dir, folder, real)

    def _mask_path(self, name):
        return self._img_path(name).replace(".tif", "_cl.tif")

    def __len__(self): return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        with rasterio.open(self._img_path(name)) as s:
            img = s.read().astype(np.float32) / 10000.0
            img = np.nan_to_num(img, nan=0., posinf=1., neginf=0.)
            img = np.clip(img, 0., 1.)
        with rasterio.open(self._mask_path(name)) as s:
            raw = s.read(1).astype(np.int32)
            mask = np.where(raw > 0, raw - 1, IGNORE_IDX).astype(np.int64)
        return (torch.tensor(img, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.long),
                name)


# ── SegFormer loader ──────────────────────────────────────────────────────────
def build_segformer(in_ch):
    from transformers import SegformerForSemanticSegmentation, SegformerConfig
    cfg = SegformerConfig(
        num_labels=NUM_CLASSES, num_channels=in_ch,
        depths=[2,2,2,2], hidden_sizes=[32,64,160,256], decoder_hidden_size=256,
    )
    model = SegformerForSemanticSegmentation(cfg)
    old = model.segformer.encoder.patch_embeddings[0].proj
    new = nn.Conv2d(in_ch, old.out_channels, old.kernel_size, old.stride, old.padding)
    nn.init.kaiming_normal_(new.weight)
    model.segformer.encoder.patch_embeddings[0].proj = new
    return model


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Scanning for Marine Debris patches in test set...")
    ds     = MaridaTestDataset(TEST_SPLIT, PATCHES_DIR)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    print(f"\nLoading SegFormer from {MODEL_PATH} ...")
    model = build_segformer(ds.in_channels).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False), strict=False)
    model.eval()

    total_debris_pixels = 0
    correct_debris_pixels = 0
    debris_inter = 0.0
    debris_union = 0.0

    per_patch = []

    with torch.no_grad():
        for imgs, masks, names in loader:
            imgs  = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            out   = model(imgs)
            logits = F.interpolate(out.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            preds  = logits.argmax(1)

            for i in range(masks.shape[0]):
                m = masks[i]
                p = preds[i]
                name = names[i]

                # Pixels that are truly Marine Debris in this patch
                gt_debris   = (m == DEBRIS_CLS)
                pred_debris = (p == DEBRIS_CLS)
                valid       = (m != IGNORE_IDX)

                n_gt = gt_debris.sum().item()
                if n_gt == 0:
                    continue

                # Per-patch metrics
                inter = (pred_debris & gt_debris).sum().item()
                union = (pred_debris | gt_debris).sum().item()
                iou   = inter / (union + 1e-9)
                tp    = inter
                fp    = (pred_debris & ~gt_debris & valid).sum().item()
                fn    = (gt_debris & ~pred_debris).sum().item()
                recall    = tp / (tp + fn + 1e-9)
                precision = tp / (tp + fp + 1e-9)

                per_patch.append((name, n_gt, iou, precision, recall))

                debris_inter += inter
                debris_union += union
                total_debris_pixels += n_gt
                correct_debris_pixels += tp

    overall_iou      = debris_inter / (debris_union + 1e-9)
    overall_recall   = correct_debris_pixels / (total_debris_pixels + 1e-9)

    print("\n" + "="*65)
    print("   FOCUSED MARINE DEBRIS EVALUATION (Debris Patches Only)")
    print("="*65)
    print(f"  Patches evaluated:           {len(per_patch)}")
    print(f"  Total debris pixels (GT):    {total_debris_pixels:,}")
    print(f"  Correctly detected pixels:   {correct_debris_pixels:,}")
    print(f"  Overall Debris IoU:          {overall_iou:.4f}  ({overall_iou*100:.1f}%)")
    print(f"  Overall Debris Recall:       {overall_recall:.4f}  ({overall_recall*100:.1f}%)")
    print("="*65)

    print(f"\n{'Patch':<35} {'GT Pixels':>10} {'IoU':>8} {'Prec':>8} {'Recall':>8}")
    print("-"*73)
    for name, n_gt, iou, prec, rec in sorted(per_patch, key=lambda x: -x[2])[:30]:
        short = name[-30:]
        print(f"{short:<35} {n_gt:>10,} {iou:>8.4f} {prec:>8.4f} {rec:>8.4f}")

    if len(per_patch) > 30:
        print(f"  ... ({len(per_patch)-30} more patches not shown)")


if __name__ == "__main__":
    main()

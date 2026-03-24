"""
SegTransformer Model Training — Anti-Overfitting & Debris-Focused
==================================================================
Trains SegTransformer on balanced MARIDA dataset with:
  • Class weighting (especially Marine Debris)
  • Focal Loss or Dice Loss
  • Validation-based (not early stopping) checkpoint selection
  • Prevents overfitting with strong regularization

Usage:
    python train_segtransformer_balanced.py \
        --data_dir D:/Plastic-Ledger/U-net-models/dataset/MARIDA_BALANCED \
        --output_dir runs/segtransformer_debris_balanced \
        --epochs 100 \
        --loss dice
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import rasterio
except ImportError:
    os.system("pip install rasterio")
    import rasterio

try:
    from transformers import SegformerForSemanticSegmentation, SegformerConfig
except ImportError:
    os.system("pip install transformers")
    from transformers import SegformerForSemanticSegmentation, SegformerConfig

try:
    import cv2
except ImportError:
    os.system("pip install opencv-python")
    import cv2

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
NUM_CLASSES = 15
NUM_BANDS = 11
IGNORE_IDX = 255

CLASS_MAP = {
    0:  "Marine Debris",        1:  "Dense Sargassum",
    2:  "Sparse Sargassum",     3:  "Natural Organic Material",
    4:  "Ship",                 5:  "Clouds",
    6:  "Marine Water",         7:  "Sediment-Laden Water",
    8:  "Foam",                 9:  "Turbid Water",
    10: "Shallow Water",        11: "Waves",
    12: "Cloud Shadows",        13: "Wakes",
    14: "Mixed Water",
}

BAND_MEANS = np.array([0.057, 0.054, 0.046, 0.036, 0.033,
                        0.041, 0.049, 0.043, 0.050, 0.031, 0.019], dtype=np.float32)
BAND_STDS  = np.array([0.010, 0.010, 0.013, 0.010, 0.012,
                        0.020, 0.030, 0.020, 0.030, 0.020, 0.013], dtype=np.float32)


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class MaridaBalancedDataset(Dataset):
    def __init__(self, splits_file, patches_dir, band_means=None, band_stds=None,
                 augment=False):
        self.patches_dir = patches_dir
        self.augment = augment
        self.band_means = band_means if band_means is not None else BAND_MEANS
        self.band_stds = band_stds if band_stds is not None else BAND_STDS
        
        with open(splits_file) as f:
            self.names = [l.strip() for l in f if l.strip()]
        
        # Get image metadata
        first_img = self._img_path(self.names[0])
        with rasterio.open(first_img) as s:
            self.in_channels = s.count
    
    def _img_path(self, name):
        parts = name.split("_")
        folder = "_".join(parts[:-1]) if "_" in name else parts[0]
        real = f"S2_{name}.tif" if not name.startswith("S2_") else f"{name}.tif"
        return self.patches_dir / folder / real
    
    def _mask_path(self, name):
        parts = name.split("_")
        folder = "_".join(parts[:-1]) if "_" in name else parts[0]
        real = f"S2_{name}_cl.tif" if not name.startswith("S2_") else f"{name}_cl.tif"
        return self.patches_dir / folder / real
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        name = self.names[idx]
        
        # Read image
        with rasterio.open(self._img_path(name)) as s:
            img = s.read().astype(np.float32) / 10000.0
            img = np.nan_to_num(img, nan=0., posinf=1., neginf=0.)
            img = np.clip(img, 0., 1.)
        
        # Read mask
        with rasterio.open(self._mask_path(name)) as s:
            raw = s.read(1).astype(np.int32)
            mask = np.where(raw > 0, raw - 1, IGNORE_IDX).astype(np.int64)
        
        # Normalize
        for b in range(min(self.in_channels, len(self.band_means))):
            img[b] = (img[b] - self.band_means[b]) / (self.band_stds[b] + 1e-6)
        
        # Augmentation (random flip/rotate)
        if self.augment and np.random.rand() > 0.5:
            img = np.flip(img, axis=-1).copy()
            mask = np.flip(mask, axis=-1).copy()
        if self.augment and np.random.rand() > 0.5:
            img = np.rot90(img, k=1, axes=(1, 2)).copy()
            mask = np.rot90(mask, k=1, axes=(0, 1)).copy()
        
        return (torch.tensor(img, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.long))


# ─────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────
class DiceLoss(nn.Module):
    """Dice (F1) loss for multi-class segmentation."""
    def __init__(self, weight=None, ignore_idx=255):
        super().__init__()
        self.weight = weight
        self.ignore_idx = ignore_idx
    
    def forward(self, logits, targets):
        # logits: (B, C, H, W), targets: (B, H, W)
        probs = F.softmax(logits, dim=1)
        
        b, c, h, w = probs.shape
        valid = targets != self.ignore_idx
        
        dice_loss = 0.0
        for cls in range(c):
            p = probs[:, cls, :, :][valid]
            t = (targets == cls)[valid].float()
            
            if t.sum() == 0:
                continue
            
            inter = (p * t).sum()
            union = p.sum() + t.sum()
            dice = 2 * inter / (union + 1e-8)
            
            loss = 1 - dice
            if self.weight is not None:
                loss *= self.weight[cls]
            
            dice_loss += loss
        
        return dice_loss / c


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    def __init__(self, weight=None, ignore_idx=255, alpha=0.25, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.ignore_idx = ignore_idx
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none',
                             ignore_index=self.ignore_idx, weight=self.weight)
        probs = torch.exp(-ce)
        focal_weight = self.alpha * (1 - probs) ** self.gamma
        return (focal_weight * ce).mean()


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def build_segtransformer(in_channels, num_classes):
    cfg = SegformerConfig(
        num_labels=num_classes,
        num_channels=in_channels,
        depths=[2, 2, 2, 2],
        hidden_sizes=[32, 64, 160, 256],
        decoder_hidden_size=256,
    )
    model = SegformerForSemanticSegmentation(cfg)
    
    # Adapt first layer for 11 input channels
    old_proj = model.segformer.encoder.patch_embeddings[0].proj
    new_proj = nn.Conv2d(in_channels, old_proj.out_channels,
                         kernel_size=old_proj.kernel_size, stride=old_proj.stride,
                         padding=old_proj.padding)
    nn.init.kaiming_normal_(new_proj.weight)
    model.segformer.encoder.patch_embeddings[0].proj = new_proj
    
    return model


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
@torch.no_grad()
def eval_step(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        out = model(imgs)
        logits = out.logits
        
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:],
                                   mode="bilinear", align_corners=False)
        
        loss = criterion(logits, masks)
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / (num_batches + 1e-9)


def compute_class_weights(data_dir):
    """Compute class weights based on training set pixel distribution."""
    train_file = data_dir / "splits" / "train_X.txt"
    patches_dir = data_dir / "patches"
    
    class_counts = defaultdict(float)
    total_pixels = 0
    
    patch_names = train_file.read_text().strip().splitlines()
    print(f"Computing class weights from {len(patch_names)} patches...")
    
    for i, name in enumerate(patch_names):
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{len(patch_names)}")
        
        name = name.strip()
        parts = name.split("_")
        folder = "_".join(parts[:-1]) if "_" in name else parts[0]
        mask_file = f"S2_{name}_cl.tif" if not name.startswith("S2_") else f"{name}_cl.tif"
        mask_path = patches_dir / folder / mask_file
        
        if not mask_path.exists():
            continue
        
        try:
            with rasterio.open(mask_path) as src:
                raw = src.read(1).astype(np.int32)
                unique, counts = np.unique(raw, return_counts=True)
                for cls, cnt in zip(unique, counts):
                    if 1 <= cls <= NUM_CLASSES:
                        class_counts[cls - 1] += cnt
                        total_pixels += cnt
        except:
            continue
    
    # Compute weights (inverse frequency, normalized)
    weights = np.ones(NUM_CLASSES, dtype=np.float32)
    for cls in range(NUM_CLASSES):
        if class_counts.get(cls, 0) > 0:
            weights[cls] = total_pixels / (NUM_CLASSES * class_counts[cls])
    
    # Extra boost for Marine Debris (class 0)
    weights[0] *= 2.0
    
    weights = weights / weights.mean()  # Normalize
    
    print(f"\nClass weights (Marine Debris boost x2):")
    for cls in range(NUM_CLASSES):
        print(f"  {CLASS_MAP[cls]:<28}: {weights[cls]:.4f}")
    
    return torch.tensor(weights, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="runs/segtransformer_balanced")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss", type=str, default="dice", choices=["ce", "dice", "focal"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    
    print(f"\n{'='*80}")
    print(f"🧠 SEGTRANSFORMER TRAINING — BALANCED MARINE DEBRIS")
    print(f"{'='*80}")
    print(f"Data:         {data_dir}")
    print(f"Output:       {output_dir}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Epochs:       {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Loss:         {args.loss}")
    print(f"Device:       {device}")
    print(f"{'='*80}\n")
    
    # Build datasets
    train_file = data_dir / "splits" / "train_X.txt"
    val_file = data_dir / "splits" / "val_X.txt"
    patches_dir = data_dir / "patches"
    
    print("Building datasets...")
    train_ds = MaridaBalancedDataset(train_file, patches_dir, augment=True)
    val_ds = MaridaBalancedDataset(val_file, patches_dir, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    
    print(f"  Train: {len(train_ds)} patches")
    print(f"  Val:   {len(val_ds)} patches\n")
    
    # Compute class weights
    class_weights = compute_class_weights(data_dir).to(device)
    
    # Build model
    print("\nBuilding SegTransformer...")
    model = build_segtransformer(train_ds.in_channels, NUM_CLASSES).to(device)
    
    # Loss function
    if args.loss == "dice":
        criterion = DiceLoss(weight=class_weights, ignore_idx=IGNORE_IDX)
    elif args.loss == "focal":
        criterion = FocalLoss(weight=class_weights, ignore_idx=IGNORE_IDX)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_IDX)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = -1
    history = {"train_loss": [], "val_loss": []}
    
    print(f"\nStarting training...\n")
    print(f"{'Epoch':<8} {'Train Loss':>12} {'Val Loss':>12} {'Best':>8}")
    print(f"{'-'*40}")
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            out = model(imgs)
            logits = out.logits
            
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:],
                                       mode="bilinear", align_corners=False)
            
            loss = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        train_loss /= (num_batches + 1e-9)
        
        # Validate
        val_loss = eval_step(model, val_loader, criterion, device)
        scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        # Checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            best_ckpt = output_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
                "class_weights": class_weights.cpu().numpy().tolist(),
                "num_bands": train_ds.in_channels,
                "num_classes": NUM_CLASSES,
                "band_means": BAND_MEANS.tolist(),
                "band_stds": BAND_STDS.tolist(),
            }, best_ckpt)
        
        best_marker = "✓" if is_best else ""
        print(f"{epoch+1:<8} {train_loss:>12.4f} {val_loss:>12.4f} {best_marker:>8}")
        
        # Early stop if no improvement for 30 epochs
        if epoch - best_epoch > 30:
            print(f"\n⏹️  No improvement for 30 epochs. Stopping.")
            break
    
    # Save final model
    final_ckpt = output_dir / "final_model.pth"
    torch.save({
        "epoch": args.epochs,
        "model_state": model.state_dict(),
        "num_bands": train_ds.in_channels,
        "num_classes": NUM_CLASSES,
    }, final_ckpt)
    
    # Save history
    hist_file = output_dir / "training_history.json"
    with open(hist_file, "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best model: {best_ckpt} (epoch {best_epoch + 1})")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"History saved: {hist_file}\n")


if __name__ == "__main__":
    main()

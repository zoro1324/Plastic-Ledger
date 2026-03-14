"""
MARIDA Marine Debris Segmentation — Training Script
=====================================================
Architecture : U-Net with ResNet-34 encoder (segmentation_models_pytorch)
Bands        : All 11 Sentinel-2 bands
Loss         : Focal Loss + Dice Loss (combined)
Optimizer    : AdamW with cosine annealing LR schedule
Augmentation : Horizontal/vertical flip, rotation, radiometric jitter

Usage:
    python marida_train.py --data_dir D:/Plastic-Ledger/models/dataset/MARIDA \
                           --output_dir runs/marida_v1 \
                           --epochs 50 \
                           --batch_size 8

Install dependencies:
    pip install torch torchvision segmentation-models-pytorch albumentations rasterio tqdm
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import rasterio
except ImportError:
    os.system("pip install rasterio")
    import rasterio

try:
    import segmentation_models_pytorch as smp
except ImportError:
    os.system("pip install segmentation-models-pytorch")
    import segmentation_models_pytorch as smp

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    os.system("pip install albumentations")
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
NUM_CLASSES = 15
NUM_BANDS   = 11
PATCH_SIZE  = 256

CLASS_MAP = {
    1: "Marine Debris",       2: "Dense Sargassum",
    3: "Sparse Sargassum",    4: "Natural Organic Material",
    5: "Ship",                6: "Clouds",
    7: "Marine Water",        8: "Sediment-Laden Water",
    9: "Foam",                10: "Turbid Water",
    11: "Shallow Water",      12: "Waves",
    13: "Cloud Shadows",      14: "Wakes",
    15: "Mixed Water",
}

# Per-band mean and std computed from dataset analysis
# Values are already in reflectance (0–1 range, ~0.02–0.06)
BAND_MEANS = np.array([0.057, 0.054, 0.046, 0.036, 0.033,
                        0.041, 0.049, 0.043, 0.050, 0.031, 0.019], dtype=np.float32)
BAND_STDS  = np.array([0.010, 0.010, 0.013, 0.010, 0.012,
                        0.020, 0.030, 0.020, 0.030, 0.020, 0.013], dtype=np.float32)

# Inverse-frequency class weights (higher = rarer class gets more weight)
# Derived from pixel distribution analysis
RAW_COUNTS = {
    1: 144,    2: 222,    3: 641,    4: 7,
    5: 728,    6: 13943,  7: 13631,  8: 8741,
    9: 282,    10: 12981, 11: 888,   12: 987,
    13: 1347,  14: 1647,  15: 43,
}


def compute_class_weights(raw_counts: dict, num_classes: int = 15) -> torch.Tensor:
    """Compute inverse-frequency weights, clipped to avoid extreme values."""
    total = sum(raw_counts.values())
    weights = []
    for cls_id in range(1, num_classes + 1):
        cnt = raw_counts.get(cls_id, 1)
        w = total / (num_classes * cnt)
        weights.append(w)
    weights = np.array(weights, dtype=np.float32)
    # Clip: max weight = 50x the median to prevent instability
    median = np.median(weights)
    # Cap at 10x median — prevents extreme weights destabilizing loss
    weights = np.clip(weights, 0, 10 * median)
    # Normalize so mean = 1
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class MARIDADataset(Dataset):
    """
    Loads MARIDA patches from the official train/val/test split files.
    Returns:
        image : (11, 256, 256) float32 tensor — normalized
        mask  : (256, 256)     int64 tensor  — class IDs 0..14 (0=background)
    """

    def __init__(self, data_dir: Path, split: str = "train",
                 augment: bool = True, ignore_background: bool = False):
        self.data_dir         = data_dir
        self.split            = split
        self.augment          = augment
        self.ignore_background = ignore_background

        # Load split file
        split_file = data_dir / "splits" / f"{split}_X.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        patch_names = split_file.read_text().strip().splitlines()
        self.pairs  = self._resolve_pairs(patch_names)
        print(f"  [{split}] {len(self.pairs)} patches loaded")

        # Augmentations
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.3),
                # Radiometric jitter — applied only to image channels
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.4
                ),
                A.GaussNoise(var_limit=(0.0001, 0.001), p=0.3),
            ])
        else:
            self.transform = None

    def _resolve_pairs(self, patch_names):
        """Find (image_path, mask_path) pairs for each patch name."""
        # Build lookup: stem → full path
        all_tifs   = list(self.data_dir.rglob("*.tif"))
        img_lookup  = {f.stem: f for f in all_tifs
                       if "_cl" not in f.stem and "_conf" not in f.stem}
        mask_lookup = {f.stem.replace("_cl", ""): f for f in all_tifs
                       if "_cl" in f.stem}

        pairs = []
        missing = 0
        for name in patch_names:
            name = name.strip()
            # Split files use "1-12-19_48MYU_0" but disk has "S2_1-12-19_48MYU_0"
            disk_name = f"S2_{name}" if not name.upper().startswith("S2_") else name
            if disk_name in img_lookup and disk_name in mask_lookup:
                pairs.append((img_lookup[disk_name], mask_lookup[disk_name]))
            else:
                missing += 1

        if missing > 0:
            print(f"  ⚠️  {missing} patches not found on disk (skipped)")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]

        # Load image (11 bands)
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)  # (11, H, W)

        # Load mask
        with rasterio.open(msk_path) as src:
            mask = src.read(1).astype(np.int32)     # (H, W)  values 1–15 or 0

        # Convert mask to 0-indexed (0=background, 1=Marine Debris, …)
        # Classes in file are 1–15; subtract 1 so they become 0–14
        # Background (unlabeled) pixels stay at 0 after we treat original 0 as background
        mask = np.where(mask > 0, mask - 1, 0).astype(np.int64)
        # Now: 0=Marine Debris, 1=Dense Sargassum, … 14=Mixed Water
        # BUT unlabeled pixels are also 0 — we need to keep original 0 as IGNORE
        # So remap: original 0→255 (ignore), 1→0, 2→1, …
        # Re-do properly:
        with rasterio.open(msk_path) as src:
            mask_raw = src.read(1).astype(np.int32)
        mask = np.where(mask_raw > 0, mask_raw - 1, 255).astype(np.int64)
        # mask: 0..14 = valid classes, 255 = ignore (background/unlabeled)

        # Normalize per-band
        for b in range(NUM_BANDS):
            image[b] = (image[b] - BAND_MEANS[b]) / (BAND_STDS[b] + 1e-6)

        # Augmentation (albumentations expects HWC)
        if self.transform is not None:
            image_hwc = np.transpose(image, (1, 2, 0))  # (H, W, 11)
            augmented = self.transform(image=image_hwc, mask=mask.astype(np.float32))
            image = np.transpose(augmented["image"], (2, 0, 1))
            mask  = augmented["mask"].astype(np.int64)

        image_tensor = torch.from_numpy(image).float()
        mask_tensor  = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor


# ─────────────────────────────────────────────
# LOSSES
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Numerically stable multi-class focal loss with class weights and ignore_index."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None,
                 ignore_index: int = 255, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma          = gamma
        self.weight         = weight
        self.ignore_index   = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = inputs.shape

        # Clamp logits to prevent inf/-inf BEFORE softmax
        inputs = torch.clamp(inputs, -30.0, 30.0)

        # Flatten
        inputs_flat  = inputs.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)

        # Mask ignore pixels
        valid = targets_flat != self.ignore_index
        inputs_flat  = inputs_flat[valid]
        targets_flat = targets_flat[valid]

        if inputs_flat.numel() == 0:
            return inputs.sum() * 0.0  # differentiable zero

        # Stable log_softmax
        log_probs = F.log_softmax(inputs_flat, dim=-1)
        probs     = log_probs.exp().clamp(min=1e-7, max=1.0)

        # Gather true-class log-prob and prob
        log_pt = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        pt     = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)

        # Focal weight — clamp pt to [eps, 1-eps] before power
        pt_clamped   = pt.clamp(min=1e-6, max=1.0 - 1e-6)
        focal_weight = (1.0 - pt_clamped).pow(self.gamma)

        # Class weights
        if self.weight is not None:
            w = self.weight.to(inputs.device)
            focal_weight = focal_weight * w[targets_flat]

        # Label smoothing: mix hard CE with uniform distribution
        if self.label_smoothing > 0:
            smooth_loss  = -log_probs.mean(dim=-1)
            hard_loss    = -log_pt
            log_pt_smooth = (1 - self.label_smoothing) * hard_loss +                              self.label_smoothing * smooth_loss
        else:
            log_pt_smooth = -log_pt

        loss = focal_weight * log_pt_smooth

        # Safety: remove any remaining NaN/Inf
        loss = loss[torch.isfinite(loss)]
        if loss.numel() == 0:
            return inputs.sum() * 0.0

        return loss.mean()


class DiceLoss(nn.Module):
    """Soft Dice loss over valid (non-ignore) pixels."""

    def __init__(self, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth       = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = inputs.shape
        probs = F.softmax(inputs, dim=1)  # (B, C, H, W)

        # One-hot encode targets, masking ignore pixels
        targets_clamped = targets.clone()
        targets_clamped[targets_clamped == self.ignore_index] = 0
        targets_onehot  = F.one_hot(targets_clamped, num_classes=C)  # (B, H, W, C)
        targets_onehot  = targets_onehot.permute(0, 3, 1, 2).float() # (B, C, H, W)

        # Zero out ignore pixels in one-hot
        ignore_mask = (targets == self.ignore_index).unsqueeze(1).expand_as(targets_onehot)
        targets_onehot[ignore_mask] = 0

        # Dice per class
        intersection = (probs * targets_onehot).sum(dim=(0, 2, 3))
        cardinality  = (probs + targets_onehot).sum(dim=(0, 2, 3))
        dice_per_cls = 1.0 - (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        return dice_per_cls.mean()


class WeightedCELoss(nn.Module):
    """Weighted Cross-Entropy — numerically stable fallback for all-ignore batches."""
    def __init__(self, weight: torch.Tensor, ignore_index: int = 255):
        super().__init__()
        self.weight       = weight
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, -30.0, 30.0)
        return F.cross_entropy(inputs, targets,
                               weight=self.weight.to(inputs.device),
                               ignore_index=self.ignore_index,
                               label_smoothing=0.05)


class CombinedLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, focal_gamma: float = 2.0,
                 alpha: float = 0.6):
        """
        Train : alpha * FocalLoss + (1-alpha) * DiceLoss
        Val   : WeightedCE (stable, handles all-ignore batches gracefully)
        """
        super().__init__()
        self.focal = FocalLoss(gamma=focal_gamma, weight=class_weights)
        self.dice  = DiceLoss()
        self.ce    = WeightedCELoss(weight=class_weights)
        self.alpha = alpha

    def forward(self, inputs, targets):
        # Check if batch has any valid (non-ignore) pixels
        valid_pixels = (targets != 255).sum().item()
        if valid_pixels == 0:
            # All-ignore batch: return 0 loss (no signal, no NaN)
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        focal_l = self.focal(inputs, targets)
        dice_l  = self.dice(inputs, targets)
        if not (torch.isfinite(focal_l) and torch.isfinite(dice_l)):
            # Fallback to stable CE if focal/dice blow up
            return self.ce(inputs, targets)
        return self.alpha * focal_l + (1 - self.alpha) * dice_l


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
class SegmentationMetrics:
    """Accumulates per-class IoU and pixel accuracy."""

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.intersection = np.zeros(self.num_classes, dtype=np.float64)
        self.union        = np.zeros(self.num_classes, dtype=np.float64)
        self.correct      = 0
        self.total        = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """preds: (B, H, W) argmax class IDs; targets: (B, H, W)"""
        preds   = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()

        valid = targets != self.ignore_index
        preds, targets = preds[valid], targets[valid]

        self.correct += (preds == targets).sum()
        self.total   += valid.sum()

        for cls in range(self.num_classes):
            pred_cls   = preds == cls
            target_cls = targets == cls
            self.intersection[cls] += (pred_cls & target_cls).sum()
            self.union[cls]        += (pred_cls | target_cls).sum()

    def iou_per_class(self):
        iou = np.where(self.union > 0,
                       self.intersection / (self.union + 1e-6),
                       np.nan)
        return iou

    def mean_iou(self):
        iou = self.iou_per_class()
        return np.nanmean(iou)

    def pixel_accuracy(self):
        return self.correct / (self.total + 1e-6)

    def debris_iou(self):
        """IoU specifically for Marine Debris class (index 0)."""
        if self.union[0] > 0:
            return self.intersection[0] / (self.union[0] + 1e-6)
        return float("nan")


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def build_model(num_classes: int = NUM_CLASSES, num_bands: int = NUM_BANDS,
                encoder: str = "resnet34") -> nn.Module:
    """
    U-Net with ResNet-34 encoder.
    Modified to accept 11-band input instead of 3.
    """
    model = smp.Unet(
        encoder_name    = encoder,
        encoder_weights = None,        # No ImageNet pretrain (wrong modality)
        in_channels     = num_bands,
        classes         = num_classes,
        activation      = None,        # Raw logits (loss handles softmax)
    )
    return model


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    metrics    = SegmentationMetrics(NUM_CLASSES)

    for images, masks in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()

        if scaler is not None:  # Mixed precision
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss   = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss   = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        loss_val = loss.item()
        if not torch.isfinite(torch.tensor(loss_val)):
            continue  # skip NaN/Inf batches
        total_loss += loss_val
        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)

    n = len(loader)
    return {
        "loss":         total_loss / n,
        "mIoU":         metrics.mean_iou(),
        "pixel_acc":    metrics.pixel_accuracy(),
        "debris_iou":   metrics.debris_iou(),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, debug_first_batch=False):
    model.eval().float()
    total_loss    = 0.0
    valid_batches = 0
    nan_batches   = 0
    metrics       = SegmentationMetrics(NUM_CLASSES)

    for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="  Val  ", leave=False)):
        images = images.to(device).float()
        masks  = masks.to(device)

        logits         = model(images)
        logits_clamped = torch.clamp(logits, -30.0, 30.0)

        # Debug first batch only
        if debug_first_batch and batch_idx == 0:
            print(f"\n  [DEBUG] images  : min={images.min():.4f} max={images.max():.4f} "
                  f"nan={torch.isnan(images).any()} inf={torch.isinf(images).any()}")
            print(f"  [DEBUG] logits   : min={logits.min():.4f} max={logits.max():.4f} "
                  f"nan={torch.isnan(logits).any()} inf={torch.isinf(logits).any()}")
            print(f"  [DEBUG] masks    : unique={torch.unique(masks).tolist()}")
            # Try each loss component separately
            try:
                focal_l = criterion.focal(logits_clamped, masks)
                print(f"  [DEBUG] focal    : {focal_l.item():.6f}")
            except Exception as e:
                print(f"  [DEBUG] focal    : ERROR {e}")
            try:
                dice_l = criterion.dice(logits_clamped, masks)
                print(f"  [DEBUG] dice     : {dice_l.item():.6f}")
            except Exception as e:
                print(f"  [DEBUG] dice     : ERROR {e}")

        loss     = criterion(logits_clamped, masks)
        loss_val = loss.item()

        if np.isfinite(loss_val):
            total_loss    += loss_val
            valid_batches += 1
        else:
            nan_batches += 1
            if nan_batches <= 2:
                print(f"\n  [WARN] NaN/Inf at val batch {batch_idx}: "
                      f"logits=[{logits.min():.2f},{logits.max():.2f}] "
                      f"mask={torch.unique(masks).tolist()}")

        preds = logits_clamped.argmax(dim=1)
        metrics.update(preds, masks)

    if nan_batches > 0:
        print(f"  [WARN] {nan_batches}/{nan_batches+valid_batches} val batches had NaN loss")

    avg_loss = total_loss / max(valid_batches, 1)
    return {
        "loss":       avg_loss,
        "mIoU":       metrics.mean_iou(),
        "pixel_acc":  metrics.pixel_accuracy(),
        "debris_iou": metrics.debris_iou(),
    }


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────
def plot_history(history: dict, output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, metric, title in zip(
        axes,
        ["loss", "mIoU", "debris_iou"],
        ["Loss", "Mean IoU", "Marine Debris IoU"]
    ):
        ax.plot(history["train"][metric], label="Train", color="#3A86FF")
        ax.plot(history["val"][metric],   label="Val",   color="#E63946")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("MARIDA Training History", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=150)
    plt.close()
    print(f"  📈 Training history → {output_dir / 'training_history.png'}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MARIDA Training Script")
    parser.add_argument("--data_dir",   type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="runs/marida_v1")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--encoder",    type=str, default="resnet34",
                        help="Encoder: resnet34 | resnet50 | efficientnet-b3")
    parser.add_argument("--no_amp",     action="store_true",
                        help="Disable mixed-precision training")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and not args.no_amp
    print(f"\n🖥  Device   : {device}")
    print(f"⚡ Mixed AMP : {use_amp}")
    print(f"📁 Data dir  : {data_dir}")
    print(f"💾 Output    : {output_dir}\n")

    # ── Datasets ──────────────────────────────
    print("Loading datasets...")
    train_ds = MARIDADataset(data_dir, split="train", augment=True)
    val_ds   = MARIDADataset(data_dir, split="val",   augment=False)
    test_ds  = MARIDADataset(data_dir, split="test",  augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────
    print(f"\nBuilding U-Net ({args.encoder} encoder, {NUM_BANDS} bands → {NUM_CLASSES} classes)...")
    model = build_model(num_classes=NUM_CLASSES, num_bands=NUM_BANDS,
                        encoder=args.encoder).to(device).float()  # always float32; AMP casts internally

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # ── Loss ──────────────────────────────────
    class_weights = compute_class_weights(RAW_COUNTS, NUM_CLASSES).to(device)
    print(f"\nClass weights (top 5 heaviest):")
    sorted_cls = sorted(range(NUM_CLASSES), key=lambda i: class_weights[i].item(), reverse=True)
    for idx in sorted_cls[:5]:
        print(f"  {CLASS_MAP[idx+1]:<28}: {class_weights[idx].item():.2f}x")

    criterion = CombinedLoss(class_weights=class_weights, focal_gamma=2.0, alpha=0.6)

    # ── Optimizer / Scheduler ─────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler    = torch.cuda.amp.GradScaler() if use_amp else None

    # ── History ───────────────────────────────
    history = {"train": {"loss": [], "mIoU": [], "pixel_acc": [], "debris_iou": []},
               "val":   {"loss": [], "mIoU": [], "pixel_acc": [], "debris_iou": []}}

    best_val_debris_iou = 0.0
    best_epoch          = 0

    print(f"\n{'='*65}")
    print(f"  TRAINING — {args.epochs} epochs")
    print(f"{'='*65}")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch:>3}/{args.epochs}  |  LR: {scheduler.get_last_lr()[0]:.2e}")

        train_metrics = train_one_epoch(model, train_loader, optimizer,
                                        criterion, device, scaler)
        val_metrics   = evaluate(model, val_loader, criterion, device,
                                     debug_first_batch=(epoch == 1))
        scheduler.step()

        # Log
        for split, metrics in [("train", train_metrics), ("val", val_metrics)]:
            for k, v in metrics.items():
                history[split][k].append(v)

        print(f"  Train → loss: {train_metrics['loss']:.4f}  "
              f"mIoU: {train_metrics['mIoU']:.4f}  "
              f"DebrisIoU: {train_metrics['debris_iou']:.4f}")
        print(f"  Val   → loss: {val_metrics['loss']:.4f}  "
              f"mIoU: {val_metrics['mIoU']:.4f}  "
              f"DebrisIoU: {val_metrics['debris_iou']:.4f}")

        # Save best model based on val mIoU (more stable signal early in training)
        # Switch to DebrisIoU once model starts learning (mIoU > 0.05)
        val_miou   = val_metrics["mIoU"]
        debris_iou = val_metrics["debris_iou"]
        save_score = debris_iou if val_miou > 0.05 else val_miou
        if not np.isnan(save_score) and save_score > best_val_debris_iou:
            best_val_debris_iou = save_score
            best_epoch          = epoch
            torch.save({
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metrics":    val_metrics,
                "band_means":     BAND_MEANS.tolist(),
                "band_stds":      BAND_STDS.tolist(),
                "num_classes":    NUM_CLASSES,
                "num_bands":      NUM_BANDS,
                "encoder":        args.encoder,
            }, output_dir / "best_model.pth")
            print(f"  ✅ Best model saved (DebrisIoU: {best_val_debris_iou:.4f})")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), output_dir / f"checkpoint_epoch{epoch}.pth")

    # ── Final Test Evaluation ──────────────────
    print(f"\n{'='*65}")
    print(f"  TEST EVALUATION  (best model from epoch {best_epoch})")
    print(f"{'='*65}")

    checkpoint = torch.load(output_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\n  Test mIoU         : {test_metrics['mIoU']:.4f}")
    print(f"  Test Debris IoU   : {test_metrics['debris_iou']:.4f}")
    print(f"  Test Pixel Acc    : {test_metrics['pixel_acc']:.4f}")

    # Per-class IoU on test set
    model.eval()
    final_metrics = SegmentationMetrics(NUM_CLASSES)
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="  Per-class IoU"):
            images = images.to(device)
            masks  = masks.to(device)
            logits = model(images)
            preds  = logits.argmax(dim=1)
            final_metrics.update(preds, masks)

    iou_per_class = final_metrics.iou_per_class()
    print(f"\n  Per-class IoU:")
    print(f"  {'Class':<28} {'IoU':>8}")
    print("  " + "─" * 38)
    for cls_id in range(NUM_CLASSES):
        name = CLASS_MAP.get(cls_id + 1, f"Class {cls_id}")
        iou  = iou_per_class[cls_id]
        marker = " ★" if cls_id == 0 else ""
        print(f"  {name:<28} {iou:>8.4f}{marker}")

    # Save results
    results = {
        "best_epoch":      best_epoch,
        "best_debris_iou": best_val_debris_iou,
        "test_metrics":    {k: float(v) for k, v in test_metrics.items()},
        "per_class_iou":   {CLASS_MAP[i+1]: float(iou_per_class[i])
                            for i in range(NUM_CLASSES)},
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  💾 Results saved → {output_dir / 'results.json'}")

    # Plot
    plot_history(history, output_dir)

    print(f"\n{'='*65}")
    print(f"  ✅ DONE — outputs in: {output_dir}/")
    print(f"     best_model.pth  — load for inference")
    print(f"     results.json    — final metrics")
    print(f"     training_history.png")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
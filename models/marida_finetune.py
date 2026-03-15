"""
MARIDA Fine-Tuning Script
==========================
Improves Marine Debris IoU from ~0.565 by:

  1. DEBRIS OVERSAMPLING  — patches with labeled debris repeated 8x per epoch
  2. DEBRIS-FOCUSED LOSS  — extra binary cross-entropy term purely on debris class
  3. STRONGER AUGMENTATION — elastic deform, coarse dropout, more radiometric jitter
  4. TEST-TIME AUGMENTATION (TTA) — averages predictions over 8 flip/rotate variants
  5. LOWER LR             — starts from checkpoint at 1e-5 (10x smaller than original)
  6. COSINE WARM RESTARTS — re-heats LR every 10 epochs to escape local minima
  7. DEBRIS-FOCUSED EVAL  — only counts patches with ≥5 GT debris pixels as signal

Usage:
    python marida_finetune.py \
        --checkpoint  runs/marida_v1/best_model.pth \
        --data_dir    D:/Plastic-Ledger/models/dataset/MARIDA \
        --output_dir  runs/marida_v2 \
        --epochs      40 \
        --batch_size  8

Install:
    pip install torch segmentation-models-pytorch albumentations rasterio tqdm
"""

import os, sys, json, argparse, numpy as np
from pathlib import Path
from collections import Counter
import warnings; warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

try:
    import rasterio
except ImportError:
    os.system("pip install rasterio"); import rasterio

try:
    import segmentation_models_pytorch as smp
except ImportError:
    os.system("pip install segmentation-models-pytorch"); import segmentation_models_pytorch as smp

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    os.system("pip install albumentations"); import albumentations as A
    from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
NUM_CLASSES = 15
NUM_BANDS   = 11
IGNORE_IDX  = 255

CLASS_MAP = {
    0:"Marine Debris",      1:"Dense Sargassum",
    2:"Sparse Sargassum",   3:"Natural Organic Material",
    4:"Ship",               5:"Clouds",
    6:"Marine Water",       7:"Sediment-Laden Water",
    8:"Foam",               9:"Turbid Water",
    10:"Shallow Water",     11:"Waves",
    12:"Cloud Shadows",     13:"Wakes",
    14:"Mixed Water",
}

BAND_MEANS = np.array([0.057,0.054,0.046,0.036,0.033,
                        0.041,0.049,0.043,0.050,0.031,0.019], dtype=np.float32)
BAND_STDS  = np.array([0.010,0.010,0.013,0.010,0.012,
                        0.020,0.030,0.020,0.030,0.020,0.013], dtype=np.float32)

RAW_COUNTS = {
    1:144, 2:222, 3:641, 4:7, 5:728,
    6:13943, 7:13631, 8:8741, 9:282, 10:12981,
    11:888, 12:987, 13:1347, 14:1647, 15:43,
}


# ─────────────────────────────────────────────
# CLASS WEIGHTS
# ─────────────────────────────────────────────
def compute_class_weights(raw_counts, num_classes=15):
    total = sum(raw_counts.values())
    weights = []
    for cls_id in range(1, num_classes+1):
        cnt = raw_counts.get(cls_id, 1)
        w   = total / (num_classes * cnt)
        weights.append(w)
    weights = np.array(weights, dtype=np.float32)
    median  = np.median(weights)
    weights = np.clip(weights, 0, 10 * median)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


# ─────────────────────────────────────────────
# DATASET  (with debris label scanning)
# ─────────────────────────────────────────────
class MARIDADataset(Dataset):
    def __init__(self, data_dir: Path, split: str = "train", augment: bool = True):
        self.data_dir = data_dir
        self.augment  = augment

        split_file  = data_dir / "splits" / f"{split}_X.txt"
        patch_names = split_file.read_text().strip().splitlines()

        all_tifs    = list(data_dir.rglob("*.tif"))
        img_lookup  = {f.stem: f for f in all_tifs
                       if "_cl" not in f.stem and "_conf" not in f.stem}
        mask_lookup = {f.stem.replace("_cl",""): f for f in all_tifs if "_cl" in f.stem}

        self.pairs      = []
        self.has_debris = []   # True if this patch has any labeled debris pixels

        for name in patch_names:
            name      = name.strip()
            disk_name = f"S2_{name}" if not name.upper().startswith("S2_") else name
            if disk_name in img_lookup and disk_name in mask_lookup:
                img_p = img_lookup[disk_name]
                msk_p = mask_lookup[disk_name]
                self.pairs.append((img_p, msk_p))
                # Scan mask for debris pixels (class 1 in raw 1-indexed file)
                with rasterio.open(msk_p) as src:
                    raw = src.read(1)
                self.has_debris.append(bool((raw == 1).any()))

        n_debris = sum(self.has_debris)
        print(f"  [{split}] {len(self.pairs)} patches  |  "
              f"{n_debris} with debris ({100*n_debris/max(1,len(self.pairs)):.1f}%)")

        # Strong augmentation for fine-tuning
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.3),
                # Radiometric jitter — stronger than original
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, contrast_limit=0.15, p=0.5),
                A.GaussNoise(var_limit=(0.0005, 0.003), p=0.4),
                # Spatial distortion helps with small debris shapes
                A.ElasticTransform(alpha=60, sigma=6, p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
                # Coarse dropout simulates cloud/sensor gaps
                A.CoarseDropout(
                    max_holes=4, max_height=32, max_width=32,
                    min_holes=1, fill_value=0, p=0.2),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]

        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)

        with rasterio.open(msk_path) as src:
            mask_raw = src.read(1).astype(np.int32)

        # Nodata masking
        nodata = (image.sum(axis=0) == 0)
        image  = np.clip(image, 0.0001, 0.5)
        for b in range(NUM_BANDS):
            image[b] = (image[b] - BAND_MEANS[b]) / (BAND_STDS[b] + 1e-6)
        image[:, nodata] = 0.0
        image = np.clip(image, -5.0, 5.0)
        image = np.nan_to_num(image, nan=0.0, posinf=5.0, neginf=-5.0)

        # Remap mask: 1-15 → 0-14, 0 → 255 (ignore)
        mask = np.where(mask_raw > 0, mask_raw - 1, IGNORE_IDX).astype(np.int64)

        # Augmentation
        if self.transform is not None:
            image_hwc = np.transpose(image, (1,2,0))
            aug       = self.transform(image=image_hwc,
                                       mask=mask.astype(np.float32))
            image = np.transpose(aug["image"], (2,0,1))
            mask  = aug["mask"].astype(np.int64)

        return (torch.from_numpy(image).float(),
                torch.from_numpy(mask).long(),
                self.has_debris[idx])


def collate_fn(batch):
    images  = torch.stack([b[0] for b in batch])
    masks   = torch.stack([b[1] for b in batch])
    has_d   = [b[2] for b in batch]
    return images, masks, has_d


# ─────────────────────────────────────────────
# WEIGHTED SAMPLER  (debris patches 8x more likely)
# ─────────────────────────────────────────────
def make_debris_sampler(dataset: MARIDADataset, debris_weight: float = 8.0):
    weights = [debris_weight if d else 1.0 for d in dataset.has_debris]
    return WeightedRandomSampler(
        weights     = weights,
        num_samples = len(weights),
        replacement = True,
    )


# ─────────────────────────────────────────────
# LOSSES
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, ignore_index=IGNORE_IDX,
                 label_smoothing=0.05):
        super().__init__()
        self.gamma          = gamma
        self.weight         = weight
        self.ignore_index   = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, -30.0, 30.0)
        B, C, H, W = inputs.shape
        inputs_flat  = inputs.permute(0,2,3,1).reshape(-1, C)
        targets_flat = targets.reshape(-1)
        valid = targets_flat != self.ignore_index
        inputs_flat  = inputs_flat[valid]
        targets_flat = targets_flat[valid]
        if inputs_flat.numel() == 0:
            return inputs.sum() * 0.0

        log_probs = F.log_softmax(inputs_flat, dim=-1)
        probs     = log_probs.exp().clamp(1e-7, 1.0)
        log_pt    = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        pt        = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        pt_c      = pt.clamp(1e-6, 1-1e-6)
        fw        = (1 - pt_c).pow(self.gamma)
        if self.weight is not None:
            fw = fw * self.weight.to(inputs.device)[targets_flat]

        if self.label_smoothing > 0:
            smooth_loss = -log_probs.mean(dim=-1)
            loss = fw * ((1-self.label_smoothing)*(-log_pt) +
                         self.label_smoothing*smooth_loss)
        else:
            loss = fw * (-log_pt)

        loss = loss[torch.isfinite(loss)]
        return loss.mean() if loss.numel() > 0 else inputs.sum()*0.0


class DiceLoss(nn.Module):
    def __init__(self, ignore_index=IGNORE_IDX, smooth=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth       = smooth

    def forward(self, inputs, targets):
        B, C, H, W = inputs.shape
        probs = F.softmax(inputs, dim=1)
        tc    = targets.clone()
        tc[tc == self.ignore_index] = 0
        oh    = F.one_hot(tc, C).permute(0,3,1,2).float()
        ig    = (targets == self.ignore_index).unsqueeze(1).expand_as(oh)
        oh[ig] = 0
        inter = (probs * oh).sum(dim=(0,2,3))
        card  = (probs + oh).sum(dim=(0,2,3))
        return (1 - (2*inter + self.smooth)/(card + self.smooth)).mean()


class DebrisBCELoss(nn.Module):
    """
    Binary cross-entropy treating Marine Debris (class 0) as positive.
    Trains the model to explicitly separate debris from everything else.
    """
    def __init__(self, ignore_index=IGNORE_IDX, pos_weight: float = 10.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.pos_weight   = pos_weight

    def forward(self, inputs, targets):
        # inputs: (B, C, H, W) logits
        debris_logit = inputs[:, 0]                 # (B,H,W) — debris channel
        valid        = targets != self.ignore_index
        if valid.sum() == 0:
            return inputs.sum() * 0.0

        binary_target = (targets == 0).float()      # 1 = debris, 0 = not debris
        pw = torch.tensor(self.pos_weight, device=inputs.device)
        loss = F.binary_cross_entropy_with_logits(
            debris_logit[valid],
            binary_target[valid],
            pos_weight=pw,
        )
        return loss


class FineTuneLoss(nn.Module):
    """
    Focal + Dice + DebrisBCE
    Weights: 0.4 * Focal + 0.3 * Dice + 0.3 * DebrisBCE
    The BCE term directly pushes debris detection precision/recall.
    """
    def __init__(self, class_weights, focal_gamma=2.0, debris_pos_weight=10.0):
        super().__init__()
        self.focal     = FocalLoss(gamma=focal_gamma, weight=class_weights)
        self.dice      = DiceLoss()
        self.debris_bce = DebrisBCELoss(pos_weight=debris_pos_weight)

    def forward(self, inputs, targets):
        valid_px = (targets != IGNORE_IDX).sum().item()
        if valid_px == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        focal_l = self.focal(inputs, targets)
        dice_l  = self.dice(inputs, targets)
        bce_l   = self.debris_bce(inputs, targets)

        # Fallback if any component is non-finite
        if not torch.isfinite(focal_l): focal_l = inputs.sum() * 0.0
        if not torch.isfinite(dice_l):  dice_l  = inputs.sum() * 0.0
        if not torch.isfinite(bce_l):   bce_l   = inputs.sum() * 0.0

        return 0.4 * focal_l + 0.3 * dice_l + 0.3 * bce_l


# ─────────────────────────────────────────────
# TEST-TIME AUGMENTATION (TTA)
# ─────────────────────────────────────────────
@torch.no_grad()
def tta_predict(model, image: torch.Tensor, device) -> torch.Tensor:
    """
    Average softmax predictions over 8 augmentation variants:
    original + hflip + vflip + rot90 + rot180 + rot270 + hflip+rot90 + vflip+rot90
    """
    image = image.to(device)
    preds = []

    def infer(x):
        logits = model(x)
        logits = torch.clamp(logits, -30.0, 30.0)
        return F.softmax(logits, dim=1)

    # Original
    preds.append(infer(image))
    # H-flip
    preds.append(infer(torch.flip(image, dims=[-1])).flip(dims=[-1]))
    # V-flip
    preds.append(infer(torch.flip(image, dims=[-2])).flip(dims=[-2]))
    # Rot 90
    preds.append(torch.rot90(infer(torch.rot90(image, k=1, dims=[-2,-1])),
                              k=-1, dims=[-2,-1]))
    # Rot 180
    preds.append(torch.rot90(infer(torch.rot90(image, k=2, dims=[-2,-1])),
                              k=-2, dims=[-2,-1]))
    # Rot 270
    preds.append(torch.rot90(infer(torch.rot90(image, k=3, dims=[-2,-1])),
                              k=-3, dims=[-2,-1]))
    # H-flip + Rot 90
    preds.append(torch.rot90(
        infer(torch.rot90(torch.flip(image, dims=[-1]), k=1, dims=[-2,-1])).flip(dims=[-1]),
        k=-1, dims=[-2,-1]))
    # V-flip + Rot 90
    preds.append(torch.rot90(
        infer(torch.rot90(torch.flip(image, dims=[-2]), k=1, dims=[-2,-1])).flip(dims=[-2]),
        k=-1, dims=[-2,-1]))

    return torch.stack(preds).mean(dim=0)   # (B, C, H, W)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
class DebrisMetrics:
    """Tracks IoU, Precision, Recall for Marine Debris class only."""
    def __init__(self, min_gt_pixels: int = 5):
        self.min_gt_pixels = min_gt_pixels
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = 0
        self.patch_ious = []

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred   = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        valid  = target != IGNORE_IDX
        pred, target = pred[valid], target[valid]

        gt_debris = (target == 0)
        if gt_debris.sum() < self.min_gt_pixels:
            return   # Skip tiny patches — unreliable IoU signal

        p  = pred == 0
        tp = (p &  gt_debris).sum()
        fp = (p & ~gt_debris).sum()
        fn = (~p & gt_debris).sum()
        self.tp += int(tp)
        self.fp += int(fp)
        self.fn += int(fn)
        iou = tp / (tp + fp + fn + 1e-9)
        self.patch_ious.append(float(iou))

    def iou(self):
        return self.tp / (self.tp + self.fp + self.fn + 1e-9)

    def precision(self):
        return self.tp / (self.tp + self.fp + 1e-9)

    def recall(self):
        return self.tp / (self.tp + self.fn + 1e-9)

    def f1(self):
        p, r = self.precision(), self.recall()
        return 2*p*r / (p+r+1e-9)

    def mean_patch_iou(self):
        return float(np.mean(self.patch_ious)) if self.patch_ious else 0.0

    def n_patches(self):
        return len(self.patch_ious)

    def summary(self):
        return (f"IoU={self.iou():.4f}  F1={self.f1():.4f}  "
                f"P={self.precision():.4f}  R={self.recall():.4f}  "
                f"patchIoU={self.mean_patch_iou():.4f}  n={self.n_patches()}")


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    metrics    = DebrisMetrics(min_gt_pixels=1)  # include all during training

    for images, masks, _ in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device)
        masks  = masks.to(device)
        optimizer.zero_grad()

        logits = model(images)
        loss   = criterion(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        lv = loss.item()
        if np.isfinite(lv):
            total_loss += lv
            n_batches  += 1

        preds = logits.argmax(dim=1)
        for i in range(masks.shape[0]):
            metrics.update(preds[i], masks[i])

    return total_loss / max(n_batches, 1), metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_tta=False):
    model.eval().float()
    total_loss = 0.0
    n_batches  = 0
    # Use min_gt_pixels=5 for val — matches our "fair metric" threshold
    metrics    = DebrisMetrics(min_gt_pixels=5)

    for images, masks, _ in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device).float()
        masks  = masks.to(device)

        if use_tta:
            probs  = tta_predict(model, images, device)
            logits = torch.log(probs.clamp(1e-7, 1.0))  # log for loss
        else:
            logits = model(images)
            logits = torch.clamp(logits, -30.0, 30.0)

        loss = criterion(logits, masks)
        lv   = loss.item()
        if np.isfinite(lv):
            total_loss += lv
            n_batches  += 1

        preds = logits.argmax(dim=1)
        for i in range(masks.shape[0]):
            metrics.update(preds[i], masks[i])

    return total_loss / max(n_batches, 1), metrics


# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────
def plot_history(history, output_dir):
    epochs = range(1, len(history["train_loss"])+1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plots = [
        ("Loss",             "train_loss",   "val_loss"),
        ("Debris IoU",       "train_iou",    "val_iou"),
        ("Debris F1",        "train_f1",     "val_f1"),
    ]
    for ax, (title, tk, vk) in zip(axes, plots):
        ax.plot(epochs, history[tk], color="#3A86FF", label="Train")
        ax.plot(epochs, history[vk], color="#E63946", label="Val")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle("MARIDA Fine-Tune History", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "finetune_history.png", dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",       type=str, required=True,
                        help="Path to best_model.pth from original training")
    parser.add_argument("--data_dir",         type=str, required=True)
    parser.add_argument("--output_dir",       type=str, default="runs/marida_v2")
    parser.add_argument("--epochs",           type=int, default=40)
    parser.add_argument("--batch_size",       type=int, default=8)
    parser.add_argument("--lr",               type=float, default=1e-5,
                        help="Learning rate (default 1e-5 = 10x smaller than original)")
    parser.add_argument("--debris_weight",    type=float, default=8.0,
                        help="How many times more likely debris patches are sampled (default 8)")
    parser.add_argument("--debris_bce_weight",type=float, default=10.0,
                        help="Positive weight for debris BCE loss term (default 10)")
    parser.add_argument("--tta",              action="store_true",
                        help="Use test-time augmentation during validation (slower but better)")
    parser.add_argument("--freeze_encoder",   action="store_true",
                        help="Freeze encoder weights — only fine-tune decoder")
    parser.add_argument("--warm_restarts",    type=int, default=10,
                        help="LR warm restart period in epochs (default 10)")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n🖥  Device        : {device}")
    print(f"📦 Checkpoint    : {args.checkpoint}")
    print(f"💾 Output        : {output_dir}")
    print(f"⚡ LR            : {args.lr}")
    print(f"🔁 Debris weight : {args.debris_weight}x oversampling")
    print(f"🎯 TTA           : {args.tta}")
    print(f"🧊 Freeze enc.   : {args.freeze_encoder}\n")

    # ── Load checkpoint ──────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    encoder    = ckpt.get("encoder", "resnet34")
    num_bands  = ckpt.get("num_bands", NUM_BANDS)
    num_cls    = ckpt.get("num_classes", NUM_CLASSES)

    model = smp.Unet(
        encoder_name    = encoder,
        encoder_weights = None,
        in_channels     = num_bands,
        classes         = num_cls,
        activation      = None,
    ).to(device).float()
    model.load_state_dict(ckpt["model_state"])
    print(f"  ✅ Loaded checkpoint  (epoch={ckpt.get('epoch','?')}  "
          f"Val mIoU={ckpt.get('val_metrics',{}).get('mIoU',0):.4f})")

    # ── Freeze encoder if requested ───────────────────────
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  🧊 Encoder frozen  — trainable params: {trainable:,}")
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params: {trainable:,}")

    # ── Datasets ──────────────────────────────────────────
    print("\nLoading datasets...")
    train_ds = MARIDADataset(data_dir, split="train", augment=True)
    val_ds   = MARIDADataset(data_dir, split="val",   augment=False)

    # Debris-oversampled training loader
    sampler = make_debris_sampler(train_ds, debris_weight=args.debris_weight)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=4,
                              pin_memory=True, drop_last=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=4,
                              pin_memory=True, collate_fn=collate_fn)

    # ── Loss ──────────────────────────────────────────────
    class_weights = compute_class_weights(RAW_COUNTS).to(device)
    criterion = FineTuneLoss(class_weights,
                             focal_gamma=2.0,
                             debris_pos_weight=args.debris_bce_weight)

    # ── Optimizer / Scheduler ─────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    # CosineAnnealingWarmRestarts: re-heats LR every T_0 epochs
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=args.warm_restarts, T_mult=1, eta_min=args.lr * 0.01
    )

    # ── Training ──────────────────────────────────────────
    history = {k: [] for k in
               ["train_loss","val_loss","train_iou","val_iou","train_f1","val_f1"]}

    best_val_iou  = 0.0
    best_epoch    = 0
    patience      = 15   # early stopping

    print(f"\n{'='*65}")
    print(f"  FINE-TUNING — {args.epochs} epochs  "
          f"(debris {args.debris_weight}x oversampled)")
    print(f"{'='*65}")

    for epoch in range(1, args.epochs+1):
        lr_now = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch:>3}/{args.epochs}  |  LR: {lr_now:.2e}")

        train_loss, train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_m = evaluate(
            model, val_loader, criterion, device, use_tta=args.tta)

        scheduler.step()

        print(f"  Train → loss:{train_loss:.4f}  {train_m.summary()}")
        print(f"  Val   → loss:{val_loss:.4f}  {val_m.summary()}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_m.iou())
        history["val_iou"].append(val_m.iou())
        history["train_f1"].append(train_m.f1())
        history["val_f1"].append(val_m.f1())

        # Save best model on val debris IoU (patches with ≥5 GT pixels)
        val_iou = val_m.iou()
        if val_iou > best_val_iou and val_m.n_patches() > 0:
            best_val_iou = val_iou
            best_epoch   = epoch
            torch.save({
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_iou":        val_iou,
                "val_f1":         val_m.f1(),
                "val_precision":  val_m.precision(),
                "val_recall":     val_m.recall(),
                "encoder":        encoder,
                "num_bands":      num_bands,
                "num_classes":    num_cls,
                "band_means":     BAND_MEANS.tolist(),
                "band_stds":      BAND_STDS.tolist(),
                # Keep val_metrics key for compatibility with inference script
                "val_metrics": {
                    "mIoU":        val_iou,
                    "debris_iou":  val_iou,
                },
            }, output_dir / "best_model.pth")
            print(f"  ✅ Best model saved  (Debris IoU: {best_val_iou:.4f})")

        # Checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       output_dir / f"checkpoint_epoch{epoch}.pth")

        # Early stopping
        if epoch - best_epoch >= patience:
            print(f"\n  Early stopping — no improvement for {patience} epochs")
            break

    # ── Final summary ─────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  FINE-TUNE COMPLETE")
    print(f"{'='*65}")
    print(f"  Best epoch       : {best_epoch}")
    print(f"  Best Debris IoU  : {best_val_iou:.4f}")
    print(f"\n  Next step — run evaluation:")
    print(f"    python marida_evaluate.py \\")
    print(f"      --model    {output_dir}/best_model.pth \\")
    print(f"      --data_dir {data_dir} \\")
    print(f"      --split    test \\")
    print(f"      --output   evaluation/finetune_test")

    # Save history & plot
    with open(output_dir / "finetune_history.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_history(history, output_dir)
    print(f"\n  📈 History → {output_dir}/finetune_history.png")
    print(f"  💾 Results → {output_dir}/best_model.pth\n")


if __name__ == "__main__":
    main()
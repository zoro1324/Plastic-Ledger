import os
import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import torch.nn.functional as F
from pathlib import Path
import json
from sklearn.metrics import jaccard_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STEP 1: CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "data_root": "models/dataset/MARIDA",           # path to your MARIDA dataset folder
    "num_classes": 16,                 # 15 classes + 1 for unlabeled (index 0, ignored)
    "batch_size": 8,
    "num_epochs": 30,
    "learning_rate": 6e-5,
    "num_bands": 11,                   # Sentinel-2 bands used in MARIDA
    "image_size": 256,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": "./checkpoints/best_model.pth",
    "confidence_threshold": 2,        # 1=High, 2=Moderate, skip 3=Low
}

os.makedirs("./checkpoints", exist_ok=True)
print(f"Using device: {CONFIG['device']}")

# ─────────────────────────────────────────────
# STEP 2: DATASET CLASS
# ─────────────────────────────────────────────
# MARIDA folder structure:
# MARIDA/
#   patches/
#     s2_10-01-17_16PCC/
#       s2_10-01-17_16PCC_patch_1.tif         ← image
#       s2_10-01-17_16PCC_patch_1_cl.tif      ← class mask
#       s2_10-01-17_16PCC_patch_1_conf.tif    ← confidence mask

class MARIDADataset(Dataset):
    def __init__(self, data_root, split="train", confidence_threshold=2):
        self.data_root = Path(data_root)
        self.confidence_threshold = confidence_threshold

        split_file = self.data_root / "splits" / f"{split}_X.txt"
        with open(split_file, "r") as f:
            self.patch_ids = [line.strip() for line in f.readlines()]

        # MARIDA band statistics for normalization (precomputed from dataset)
        self.band_means = np.array([
            0.0582, 0.0577, 0.0573, 0.0529, 0.0660,
            0.0757, 0.0795, 0.0757, 0.0848, 0.0018, 0.0490
        ])
        self.band_stds = np.array([
            0.0359, 0.0361, 0.0372, 0.0396, 0.0479,
            0.0543, 0.0578, 0.0556, 0.0614, 0.0015, 0.0399
        ])

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        patch_id = self.patch_ids[idx]

        # ── Build correct folder path ──
        # Split file entries look like: "20-4-18_30VWH_8"
        # Actual folder on disk looks like: "patches/S2_20-4-18_30VWH/"
        # File on disk looks like: "S2_20-4-18_30VWH_8.tif"
        # So folder = everything except the last "_N" (patch number)
        parts = patch_id.split("_")          # ["20-4-18", "30VWH", "8"]
        scene_name = "_".join(parts[:-1])    # "20-4-18_30VWH"
        folder = self.data_root / "patches" / f"S2_{scene_name}"

        # ── Load the multi-band satellite image ──
        # Each .tif file has 11 Sentinel-2 bands stacked together
        # rasterio reads it as shape (11, 256, 256)
        img_path = folder / f"S2_{patch_id}.tif"
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)        # shape: (11, 256, 256)
            # Remove NaNs which disrupt training
            image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Load the class mask ──
        # Each pixel has a value 0–15 representing the class
        # 1=Marine Debris, 2-15=Background, 0=Unlabeled
        mask_path = folder / f"S2_{patch_id}_cl.tif"
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)          # shape: (256, 256)

        # ── Load confidence mask ──
        # Pixels with confidence=3 (Low) are unreliable — we ignore them
        conf_path = folder / f"S2_{patch_id}_conf.tif"
        with rasterio.open(conf_path) as src:
            conf = src.read(1).astype(np.int64)          # shape: (256, 256)

        # ── Normalize the image bands ──
        # Subtract mean and divide by std for each band (like ImageNet normalization)
        # This helps the model converge faster
        for b in range(image.shape[0]):
            image[b] = (image[b] - self.band_means[b]) / (self.band_stds[b] + 1e-9)

        # ── Handle Mask Classes ──
        # We now keep all 15 classes separate, which improves model learning
        # Class 1 = Marine Debris, 2-15 = Background, 0 = Unlabeled
        binary_mask = mask.astype(np.int64)
        binary_mask[mask == 0] = -100

        # ── Ignore low-confidence pixels ──
        # Set low-confidence pixels to -100 so PyTorch loss ignores them
        binary_mask[conf > self.confidence_threshold] = -100

        image_tensor = torch.tensor(image, dtype=torch.float32)
        mask_tensor = torch.tensor(binary_mask, dtype=torch.long)

        return image_tensor, mask_tensor


# ─────────────────────────────────────────────
# STEP 3: BUILD THE MODEL
# ─────────────────────────────────────────────
# SegFormer is a transformer-based segmentation model
# We load the pretrained mit-b2 backbone but CHANGE the first layer
# because the original model expects 3 channels (RGB)
# but our satellite images have 11 channels

def build_model(num_classes=2, num_bands=11):
    config = SegformerConfig.from_pretrained(
        "nvidia/mit-b2",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    model = SegformerForSemanticSegmentation(config)

    # Replace the first patch embedding to accept 11 bands instead of 3
    # Original: Conv2d(3, 64, kernel_size=7, stride=4, padding=3)
    # New:      Conv2d(11, 64, kernel_size=7, stride=4, padding=3)
    old_proj = model.segformer.encoder.patch_embeddings[0].proj
    new_proj = nn.Conv2d(
        num_bands,
        old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
    )

    # Copy the weights of the first 3 channels from pretrained model
    # Initialize the remaining 8 channels randomly
    with torch.no_grad():
        new_proj.weight[:, :3, :, :] = old_proj.weight[:, :3, :, :]
        nn.init.kaiming_normal_(new_proj.weight[:, 3:, :, :])

    model.segformer.encoder.patch_embeddings[0].proj = new_proj

    return model


# ─────────────────────────────────────────────
# STEP 4: LOSS FUNCTION
# ─────────────────────────────────────────────
# Marine debris pixels are RARE compared to water pixels
# This creates class imbalance → model learns to predict "water" always
# Solution: use weighted cross-entropy loss
# We give more weight (10x) to the rare debris class

def get_loss_fn():
    # Weight index 0 = Ignored, index 1 = Debris, 2-15 = other classes
    class_weights = torch.ones(16).to(CONFIG["device"])
    class_weights[1] = 10.0  # Give more weight to the rare debris class
    return nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)


# ─────────────────────────────────────────────
# STEP 5: EVALUATION METRIC
# ─────────────────────────────────────────────
# We use IoU (Intersection over Union) also called Jaccard Score
# IoU = (Predicted ∩ Ground Truth) / (Predicted ∪ Ground Truth)
# IoU = 1.0 means perfect prediction
# IoU = 0.0 means completely wrong

def compute_iou(preds, labels):
    preds = preds.cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()

    valid = labels != -100
    preds = preds[valid]
    labels = labels[valid]

    # We are only interested in measuring IoU for the Marine Debris class (class 1)
    preds_binary = (preds == 1).astype(int)
    labels_binary = (labels == 1).astype(int)

    return jaccard_score(labels_binary, preds_binary, average="binary", zero_division=0)


# ─────────────────────────────────────────────
# STEP 6: TRAINING LOOP
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    total_iou = 0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)     # shape: (B, 11, 256, 256)
        masks = masks.to(device)       # shape: (B, 256, 256)

        # ── Forward pass ──
        # SegFormer returns logits of shape (B, num_classes, H/4, W/4)
        # The output is 4x smaller than input due to the patch embedding stride
        outputs = model(pixel_values=images)
        logits = outputs.logits        # shape: (B, 2, 64, 64)

        # ── Upsample logits back to original mask size ──
        # We need to compare with (256, 256) masks, so we resize
        logits_upsampled = F.interpolate(
            logits,
            size=(CONFIG["image_size"], CONFIG["image_size"]),
            mode="bilinear",
            align_corners=False
        )                              # shape: (B, 2, 256, 256)

        # ── Compute loss ──
        loss = loss_fn(logits_upsampled, masks)

        # ── Backward pass ──
        # Zero gradients → compute gradients → update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── Get predicted class per pixel ──
        preds = logits_upsampled.argmax(dim=1)   # shape: (B, 256, 256)

        total_loss += loss.item()
        total_iou += compute_iou(preds, masks)

        if batch_idx % 20 == 0:
            print(f"  Batch [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    avg_iou = total_iou / len(loader)
    return avg_loss, avg_iou


def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    total_iou = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(pixel_values=images)
            logits = outputs.logits

            logits_upsampled = F.interpolate(
                logits,
                size=(CONFIG["image_size"], CONFIG["image_size"]),
                mode="bilinear",
                align_corners=False
            )

            loss = loss_fn(logits_upsampled, masks)
            preds = logits_upsampled.argmax(dim=1)

            total_loss += loss.item()
            total_iou += compute_iou(preds, masks)

    return total_loss / len(loader), total_iou / len(loader)


# ─────────────────────────────────────────────
# STEP 7: MAIN — PUT IT ALL TOGETHER
# ─────────────────────────────────────────────

def main():
    # Load datasets
    print("Loading datasets...")
    train_dataset = MARIDADataset(CONFIG["data_root"], split="train", confidence_threshold=CONFIG["confidence_threshold"])
    val_dataset   = MARIDADataset(CONFIG["data_root"], split="val",   confidence_threshold=CONFIG["confidence_threshold"])
    test_dataset  = MARIDADataset(CONFIG["data_root"], split="test",  confidence_threshold=CONFIG["confidence_threshold"])

    # num_workers=0 on Windows — multiprocessing causes repeated init + crashes
    num_workers = 0 if os.name == "nt" else 4
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=CONFIG["batch_size"], shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # Build model
    print("Building model...")
    model = build_model(num_classes=CONFIG["num_classes"], num_bands=CONFIG["num_bands"])
    model = model.to(CONFIG["device"])

    # Optimizer and loss
    # AdamW is Adam with weight decay → prevents overfitting
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=0.01)

    # Learning rate scheduler: reduce LR when validation loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    loss_fn = get_loss_fn()

    # Training loop
    best_val_iou = 0.0
    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}

    print(f"\nStarting training for {CONFIG['num_epochs']} epochs...")
    for epoch in range(CONFIG["num_epochs"]):
        print(f"\nEpoch [{epoch+1}/{CONFIG['num_epochs']}]")

        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, loss_fn, CONFIG["device"])
        val_loss, val_iou     = validate(model, val_loader, loss_fn, CONFIG["device"])

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)

        print(f"  Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}   | Val   IoU: {val_iou:.4f}")

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), CONFIG["save_path"])
            print(f"  ✅ Best model saved! Val IoU: {val_iou:.4f}")

    # Final test evaluation
    print("\nLoading best model for final test evaluation...")
    model.load_state_dict(torch.load(CONFIG["save_path"]))
    test_loss, test_iou = validate(model, test_loader, loss_fn, CONFIG["device"])
    print(f"\n🎯 Final Test IoU: {test_iou:.4f}")
    print(f"🎯 Final Test Loss: {test_loss:.4f}")

    with open("./checkpoints/history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete! Check ./checkpoints/ for saved model and history.")


if __name__ == "__main__":
    main()

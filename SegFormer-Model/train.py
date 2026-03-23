import os
import torch
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from dataset import MaridaDataset
import warnings
import numpy as np
warnings.filterwarnings("ignore")

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

class SegmentationMetrics:
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
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = np.where(self.union > 0, self.intersection / self.union, np.nan)
        return iou

    def mean_iou(self):
        return np.nanmean(self.iou_per_class())

    def pixel_accuracy(self):
        return self.correct / (self.total + 1e-6)

def main():
    dataset_dir = r"d:\Plastic-Ledger\U-net-models\dataset\MARIDA"
    patches_dir = os.path.join(dataset_dir, "patches")
    splits_dir = os.path.join(dataset_dir, "splits")
    
    print("Loading MARIDA Segformer Dataset...")
    train_dataset = MaridaDataset(os.path.join(splits_dir, "train_X.txt"), patches_dir)
    val_dataset = MaridaDataset(os.path.join(splits_dir, "val_X.txt"), patches_dir)
    test_dataset = MaridaDataset(os.path.join(splits_dir, "test_X.txt"), patches_dir)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Input channels detected: {train_dataset.in_channels}")
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Initialize Model from config to accept N channels instead of 3
    num_classes = 15  # Reverted back to purely 15 classes since background 0 is now correctly ignored as 255
    config = SegformerConfig(
        num_labels=num_classes,
        num_channels=train_dataset.in_channels,
        ignore_mismatched_sizes=True
    )
    model = SegformerForSemanticSegmentation(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model successfully loaded on {device}. Dynamic Band embedding initialized.")
    
    # Official Inverse-frequency weights for the 15 MARIDA classes to combat heavy class imbalance
    # Capped at 50.0 to prevent gradient explosions on ultra-rare pixels like Organic Material
    weights_np = np.array([
        38.08, 24.71, 8.56, 50.0, 7.53, 0.39, 0.40, 0.62, 19.45, 0.42, 6.17, 5.55, 4.07, 3.32, 50.0
    ], dtype=np.float32)
    class_weights = torch.tensor(weights_np).to(device)
    
    # Override SegFormer's default un-weighted loss with our heavily weighted custom loss
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    
    # Lowered learning rate to prevent exploding gradients from the new 11-channel embedding
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)
    
    print("Starting Full Training Loop...")
    
    num_epochs = 50  # Adjust this higher (e.g. 50) for production
    os.makedirs(r"d:\Plastic-Ledger\models\runs\segformer_v1", exist_ok=True)
    best_val_loss = float('inf')
    best_model_path = r"d:\Plastic-Ledger\models\runs\segformer_v1\best_model.pth"
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            # Do not pass labels to model so we can calculate our custom weighted loss instead
            outputs = model(pixel_values=pixel_values)
            
            # SegFormer produces 1/4 resolution logits, so we must interpolate them to match the target mask exactly
            logits_resized = torch.nn.functional.interpolate(
                outputs.logits, 
                size=labels.shape[-2:], 
                mode="bilinear", 
                align_corners=False
            )
            
            loss = loss_fn(logits_resized, labels)
            
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at batch {batch_idx}. Skipping optimization step.")
                continue
                
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping is explicitly added to prevent exploding gradients (NaN losses)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
                
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss = 0
        val_metrics = SegmentationMetrics(num_classes)
        
        with torch.no_grad():
            for batch in val_loader:
                val_pixel_values = batch["pixel_values"].to(device)
                val_labels = batch["labels"].to(device)
                val_outputs = model(pixel_values=val_pixel_values, labels=val_labels)
                val_loss += val_outputs.loss.item()
                
                # Resize Segformer 1/4th logits to match label sizing for accurate pixel validation
                logits_resized = torch.nn.functional.interpolate(val_outputs.logits, size=val_labels.shape[-2:], mode="bilinear", align_corners=False)
                preds = logits_resized.argmax(dim=1)
                val_metrics.update(preds, val_labels)
                
        avg_val_loss = val_loss / len(val_loader)
        val_mIoU = val_metrics.mean_iou()
        print(f"=== Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val mIoU: {val_mIoU:.4f} ===")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f" -> Saved NEW Best Model (Val Loss: {best_val_loss:.4f}) to {best_model_path}")

    print("\nTraining Complete!")
    
    # --- TEST SET EVALUATION ---
    print("\nEvaluating Best Model on Test Set...")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()
    
    test_loss = 0
    test_metrics = SegmentationMetrics(num_classes)
    
    with torch.no_grad():
        for batch in test_loader:
            test_pixel_values = batch["pixel_values"].to(device)
            test_labels = batch["labels"].to(device)
            
            outputs = model(pixel_values=test_pixel_values, labels=test_labels)
            test_loss += outputs.loss.item()
            
            logits_resized = torch.nn.functional.interpolate(outputs.logits, size=test_labels.shape[-2:], mode="bilinear", align_corners=False)
            preds = logits_resized.argmax(dim=1)
            test_metrics.update(preds, test_labels)
            
    avg_test_loss = test_loss / len(test_loader)
    test_acc = test_metrics.pixel_accuracy() * 100
    test_mIoU = test_metrics.mean_iou()
    
    print(f"\nFinal Test Loss: {avg_test_loss:.4f}")
    print(f"Final Test Pixel Accuracy (Excl. Bg): {test_acc:.2f}%")
    print(f"Final Test Mean IoU: {test_mIoU:.4f}\n")
    
    print("--- Per-class IoU ---")
    print(f"{'Class':<28} {'IoU':>8}")
    print("-" * 38)
    
    iou_per_class = test_metrics.iou_per_class()
    for cls_id in range(num_classes): # Loop properly through all 15 target foreground classes
        name = CLASS_MAP.get(cls_id + 1, f"Class {cls_id}")
        iou = iou_per_class[cls_id]
        print(f"{name:<28} {iou:>8.4f}")

if __name__ == "__main__":
    main()

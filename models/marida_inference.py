"""
MARIDA Marine Debris — Inference Script
========================================
Runs the trained model on any Sentinel-2 GeoTIFF patch and produces:
  • Colour-coded class mask overlay
  • Marine Debris probability heatmap
  • GeoTIFF output mask (for GIS use)
  • Summary of detected classes

Usage:
    # Single patch
    python marida_inference.py \
        --model   runs/marida_v2/best_model.pth \
        --input   path/to/patch.tif \
        --output  output/

    # Entire folder of patches
    python marida_inference.py \
        --model   runs/marida_v2/best_model.pth \
        --input   D:/Plastic-Ledger/models/dataset/MARIDA/patches/S2_12-12-20_16PCC \
        --output  output/ \
        --batch

    # Show only debris detections above a threshold
    python marida_inference.py \
        --model   runs/marida_v2/best_model.pth \
        --input   path/to/patch.tif \
        --output  output/ \
        --debris_threshold 0.3
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
except ImportError:
    os.system("pip install rasterio")
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

try:
    import segmentation_models_pytorch as smp
except ImportError:
    os.system("pip install segmentation-models-pytorch")
    import segmentation_models_pytorch as smp

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
NUM_CLASSES = 15
NUM_BANDS   = 11

CLASS_MAP = {
    0:  "Marine Debris",
    1:  "Dense Sargassum",
    2:  "Sparse Sargassum",
    3:  "Natural Organic Material",
    4:  "Ship",
    5:  "Clouds",
    6:  "Marine Water",
    7:  "Sediment-Laden Water",
    8:  "Foam",
    9:  "Turbid Water",
    10: "Shallow Water",
    11: "Waves",
    12: "Cloud Shadows",
    13: "Wakes",
    14: "Mixed Water",
}

CLASS_COLORS = [
    "#E63946",  # 0  Marine Debris       — red
    "#2A9D8F",  # 1  Dense Sargassum     — teal
    "#57CC99",  # 2  Sparse Sargassum    — green
    "#F4A261",  # 3  Natural Organic     — orange
    "#264653",  # 4  Ship                — dark teal
    "#A8DADC",  # 5  Clouds              — light blue
    "#1D3557",  # 6  Marine Water        — navy
    "#E9C46A",  # 7  Sediment Water      — gold
    "#F0EFEB",  # 8  Foam                — white
    "#8338EC",  # 9  Turbid Water        — purple
    "#3A86FF",  # 10 Shallow Water       — blue
    "#FFFFFF",  # 11 Waves               — white
    "#6B6570",  # 12 Cloud Shadows       — grey
    "#B5D5C5",  # 13 Wakes               — mint
    "#80B3FF",  # 14 Mixed Water         — light blue
]

BAND_MEANS = np.array([0.057, 0.054, 0.046, 0.036, 0.033,
                        0.041, 0.049, 0.043, 0.050, 0.031, 0.019], dtype=np.float32)
BAND_STDS  = np.array([0.010, 0.010, 0.013, 0.010, 0.012,
                        0.020, 0.030, 0.020, 0.030, 0.020, 0.013], dtype=np.float32)


# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────
def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    encoder    = ckpt.get("encoder",     "resnet34")
    num_bands  = ckpt.get("num_bands",   NUM_BANDS)
    num_cls    = ckpt.get("num_classes", NUM_CLASSES)

    model = smp.Unet(
        encoder_name    = encoder,
        encoder_weights = None,
        in_channels     = num_bands,
        classes         = num_cls,
        activation      = None,
    ).to(device).float()

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Override band stats if saved in checkpoint
    band_means = np.array(ckpt.get("band_means", BAND_MEANS), dtype=np.float32)
    band_stds  = np.array(ckpt.get("band_stds",  BAND_STDS),  dtype=np.float32)

    print(f"  ✅ Loaded model (encoder={encoder}, epoch={ckpt.get('epoch','?')})")
    val_m = ckpt.get("val_metrics", {})
    if val_m:
        print(f"     Val mIoU={val_m.get('mIoU', '?'):.4f}  "
              f"DebrisIoU={val_m.get('debris_iou', '?'):.4f}")

    return model, band_means, band_stds


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_patch(tif_path: Path, band_means, band_stds):
    with rasterio.open(tif_path) as src:
        image    = src.read().astype(np.float32)  # (B, H, W)
        profile  = src.profile
        transform = src.transform
        crs      = src.crs

    n_bands = image.shape[0]
    if n_bands < NUM_BANDS:
        # Pad with zeros if fewer bands
        pad = np.zeros((NUM_BANDS - n_bands, *image.shape[1:]), dtype=np.float32)
        image = np.concatenate([image, pad], axis=0)
    elif n_bands > NUM_BANDS:
        image = image[:NUM_BANDS]

    # Nodata mask (all-zero pixels)
    nodata_mask = (image.sum(axis=0) == 0)

    # Clip and normalize
    image = np.clip(image, 0.0001, 0.5)
    for b in range(NUM_BANDS):
        image[b] = (image[b] - band_means[b]) / (band_stds[b] + 1e-6)

    image[:, nodata_mask] = 0.0
    image = np.clip(image, -5.0, 5.0)
    image = np.nan_to_num(image, nan=0.0, posinf=5.0, neginf=-5.0)

    return image, profile, transform, crs, nodata_mask


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, image_np: np.ndarray, device: torch.device,
                  debris_threshold: float = 0.0):
    """
    Returns:
        pred_mask   : (H, W)      int array of class IDs 0–14
        prob_map    : (C, H, W)   softmax probabilities
        debris_prob : (H, W)      probability of Marine Debris (class 0)
        debris_mask : (H, W)      bool — True where debris prob > threshold
    """
    tensor = torch.from_numpy(image_np).unsqueeze(0).float().to(device)  # (1,11,H,W)
    logits = model(tensor)                                                 # (1,15,H,W)
    logits = torch.clamp(logits, -30.0, 30.0)
    probs  = F.softmax(logits, dim=1)[0].cpu().numpy()                    # (15,H,W)

    pred_mask   = probs.argmax(axis=0)                                    # (H,W)
    debris_prob = probs[0]                                                 # (H,W)

    if debris_threshold > 0:
        debris_mask = debris_prob > debris_threshold
    else:
        debris_mask = pred_mask == 0

    return pred_mask, probs, debris_prob, debris_mask


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────
def make_rgb(image_np: np.ndarray) -> np.ndarray:
    """Create a percentile-stretched RGB from bands 3,2,1 (0-indexed 2,1,0)."""
    def stretch(x):
        lo, hi = np.percentile(x[x > -4.9], 2), np.percentile(x[x > -4.9], 98)
        return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    r = stretch(image_np[2])
    g = stretch(image_np[1])
    b = stretch(image_np[0])
    return np.stack([r, g, b], axis=-1)


def hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def colorize_mask(pred_mask: np.ndarray) -> np.ndarray:
    h, w = pred_mask.shape
    colored = np.zeros((h, w, 3), dtype=np.float32)
    for cls_id in range(NUM_CLASSES):
        colored[pred_mask == cls_id] = hex_to_rgb(CLASS_COLORS[cls_id])
    return colored


def save_visualization(image_np, pred_mask, debris_prob, debris_mask,
                        tif_path: Path, output_dir: Path,
                        debris_threshold: float = 0.0):
    rgb          = make_rgb(image_np)
    colored_mask = colorize_mask(pred_mask)

    present_classes = np.unique(pred_mask)
    legend_patches  = [
        mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_MAP.get(c, str(c)))
        for c in present_classes
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.patch.set_facecolor("#0d1117")

    titles = ["RGB Composite", "Predicted Classes",
              "Marine Debris Probability", "Debris Detection"]
    for ax, title in zip(axes, titles):
        ax.set_facecolor("#0d1117")
        ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=8)
        ax.axis("off")

    # Panel 1: RGB
    axes[0].imshow(rgb)

    # Panel 2: Class mask
    axes[1].imshow(colored_mask)
    axes[1].legend(handles=legend_patches, loc="lower right",
                   fontsize=6, framealpha=0.8,
                   facecolor="#1a1a2e", labelcolor="white")

    # Panel 3: Debris probability heatmap
    im = axes[2].imshow(debris_prob, cmap="hot", vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04).ax.tick_params(colors="white")

    # Panel 4: Binary debris mask overlaid on RGB
    axes[3].imshow(rgb)
    if debris_mask.any():
        overlay = np.zeros((*debris_mask.shape, 4), dtype=np.float32)
        overlay[debris_mask] = [0.9, 0.1, 0.1, 0.7]  # red with alpha
        axes[3].imshow(overlay)
        debris_pct = 100 * debris_mask.sum() / debris_mask.size
        axes[3].set_title(
            f"Debris Detection  ({debris_pct:.2f}% pixels)",
            color="#E63946", fontsize=10, fontweight="bold", pad=8
        )
    else:
        axes[3].set_title("Debris Detection  (none detected)",
                          color="gray", fontsize=10, fontweight="bold", pad=8)

    plt.suptitle(f"MARIDA Inference — {tif_path.name}",
                 color="white", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = output_dir / f"{tif_path.stem}_inference.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    plt.close()
    return out_path


# ─────────────────────────────────────────────
# GEOTIFF OUTPUT
# ─────────────────────────────────────────────
def save_geotiff_mask(pred_mask: np.ndarray, debris_prob: np.ndarray,
                       profile, transform, crs, output_dir: Path,
                       tif_path: Path):
    """Save predicted class mask + debris probability as a 2-band GeoTIFF."""
    out_path = output_dir / f"{tif_path.stem}_mask.tif"

    profile.update(
        count=2,
        dtype=rasterio.uint8,
        compress="lzw",
        nodata=255,
    )
    # Band 1: class IDs (uint8), Band 2: debris prob scaled to 0–100
    mask_u8   = pred_mask.astype(np.uint8)
    debris_u8 = (debris_prob * 100).clip(0, 100).astype(np.uint8)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask_u8,   1)
        dst.write(debris_u8, 2)
        dst.update_tags(
            1,
            description="Predicted class ID (0=Marine Debris … 14=Mixed Water)"
        )
        dst.update_tags(
            2,
            description="Marine Debris probability x100 (0–100)"
        )

    return out_path


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
def print_summary(pred_mask, debris_prob, debris_mask, tif_path):
    print(f"\n  📍 {tif_path.name}")
    total_px = pred_mask.size

    present = np.unique(pred_mask)
    print(f"  {'Class':<28} {'Pixels':>8} {'%':>7}")
    print("  " + "─" * 46)
    for cls_id in present:
        name = CLASS_MAP.get(cls_id, f"Class {cls_id}")
        cnt  = (pred_mask == cls_id).sum()
        pct  = 100 * cnt / total_px
        marker = " ★" if cls_id == 0 else ""
        print(f"  {name:<28} {cnt:>8,} {pct:>6.2f}%{marker}")

    debris_pct = 100 * debris_mask.sum() / total_px
    avg_prob   = debris_prob[debris_mask].mean() if debris_mask.any() else 0.0
    print(f"\n  🗑  Debris pixels detected : {debris_mask.sum():,} ({debris_pct:.3f}%)")
    print(f"  🎯 Avg debris confidence  : {avg_prob:.3f}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MARIDA Inference Script")
    parser.add_argument("--model",  type=str, required=True,
                        help="Path to best_model.pth checkpoint")
    parser.add_argument("--input",  type=str, required=True,
                        help="Path to a .tif patch OR a folder of .tif patches")
    parser.add_argument("--output", type=str, default="inference_output",
                        help="Output directory")
    parser.add_argument("--batch",  action="store_true",
                        help="Process all .tif files in --input folder")
    parser.add_argument("--debris_threshold", type=float, default=0.0,
                        help="Probability threshold for debris detection (0=argmax, e.g. 0.3)")
    parser.add_argument("--save_geotiff", action="store_true",
                        help="Also save GeoTIFF mask output")
    parser.add_argument("--no_viz", action="store_true",
                        help="Skip visualization (faster for batch processing)")
    args = parser.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🖥  Device : {device}")
    print(f"📦 Model  : {args.model}")
    print(f"💾 Output : {output_dir}\n")

    # Load model
    model, band_means, band_stds = load_model(args.model, device)

    # Collect input files
    input_path = Path(args.input)
    if args.batch or input_path.is_dir():
        tif_files = sorted([
            f for f in input_path.rglob("*.tif")
            if "_cl" not in f.stem and "_conf" not in f.stem
        ])
        print(f"  Found {len(tif_files)} patches in {input_path}")
    else:
        tif_files = [input_path]

    if not tif_files:
        print("❌ No .tif files found.")
        sys.exit(1)

    # Run inference
    print(f"\n{'='*55}")
    print(f"  RUNNING INFERENCE ON {len(tif_files)} PATCH(ES)")
    print(f"{'='*55}")

    debris_summary = []

    for tif_path in tif_files:
        try:
            # Preprocess
            image_np, profile, transform, crs, nodata_mask = preprocess_patch(
                tif_path, band_means, band_stds
            )

            # Infer
            pred_mask, probs, debris_prob, debris_mask = run_inference(
                model, image_np, device, args.debris_threshold
            )

            # Mask nodata pixels as background
            pred_mask[nodata_mask] = 6   # Marine Water (most likely background)

            # Print summary
            print_summary(pred_mask, debris_prob, debris_mask, tif_path)

            # Visualise
            if not args.no_viz:
                viz_path = save_visualization(
                    image_np, pred_mask, debris_prob, debris_mask,
                    tif_path, output_dir, args.debris_threshold
                )
                print(f"  📊 Visualization → {viz_path}")

            # GeoTIFF
            if args.save_geotiff:
                gtiff_path = save_geotiff_mask(
                    pred_mask, debris_prob, profile, transform,
                    crs, output_dir, tif_path
                )
                print(f"  🗺  GeoTIFF mask  → {gtiff_path}")

            debris_pct = 100 * debris_mask.sum() / pred_mask.size
            debris_summary.append({
                "patch":       tif_path.name,
                "debris_pct":  debris_pct,
                "debris_px":   int(debris_mask.sum()),
                "avg_conf":    float(debris_prob[debris_mask].mean()) if debris_mask.any() else 0.0,
            })

        except Exception as e:
            print(f"  ⚠️  Skipping {tif_path.name}: {e}")
            continue

    # Batch summary
    if len(debris_summary) > 1:
        print(f"\n{'='*55}")
        print(f"  BATCH SUMMARY — Top debris patches")
        print(f"{'='*55}")
        top = sorted(debris_summary, key=lambda x: x["debris_pct"], reverse=True)[:10]
        print(f"  {'Patch':<40} {'Debris%':>8} {'AvgConf':>8}")
        print("  " + "─" * 58)
        for r in top:
            print(f"  {r['patch']:<40} {r['debris_pct']:>7.3f}%  {r['avg_conf']:>7.3f}")

    print(f"\n✅ Done — outputs in: {output_dir}/\n")


if __name__ == "__main__":
    main()
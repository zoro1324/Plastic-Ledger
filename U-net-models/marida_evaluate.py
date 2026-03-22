"""
MARIDA Batch Evaluation — Test Set Report
==========================================
Runs inference on all test patches and produces:
  • Per-patch metrics (IoU, F1, Precision, Recall)
  • Aggregate metrics across all patches with GT debris
  • Summary CSV
  • Visual report chart

Usage:
    python marida_evaluate.py \
        --model   runs/marida_v1/best_model.pth \
        --data_dir D:/Plastic-Ledger/models/dataset/MARIDA \
        --split   test \
        --output  evaluation/
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

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
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
NUM_CLASSES = 15
NUM_BANDS   = 11

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

CLASS_COLORS = [
    "#E63946", "#2A9D8F", "#57CC99", "#F4A261", "#264653",
    "#A8DADC", "#1D3557", "#E9C46A", "#F0EFEB", "#8338EC",
    "#3A86FF", "#FFFFFF", "#6B6570", "#B5D5C5", "#80B3FF",
]

BAND_MEANS = np.array([0.057, 0.054, 0.046, 0.036, 0.033,
                        0.041, 0.049, 0.043, 0.050, 0.031, 0.019], dtype=np.float32)
BAND_STDS  = np.array([0.010, 0.010, 0.013, 0.010, 0.012,
                        0.020, 0.030, 0.020, 0.030, 0.020, 0.013], dtype=np.float32)


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = smp.Unet(
        encoder_name    = ckpt.get("encoder", "resnet34"),
        encoder_weights = None,
        in_channels     = ckpt.get("num_bands", NUM_BANDS),
        classes         = ckpt.get("num_classes", NUM_CLASSES),
        activation      = None,
    ).to(device).float()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    band_means = np.array(ckpt.get("band_means", BAND_MEANS), dtype=np.float32)
    band_stds  = np.array(ckpt.get("band_stds",  BAND_STDS),  dtype=np.float32)
    print(f"  ✅ Model loaded (epoch={ckpt.get('epoch','?')}, "
          f"Val mIoU={ckpt.get('val_metrics',{}).get('mIoU',0):.4f})")
    return model, band_means, band_stds


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(tif_path, band_means, band_stds):
    with rasterio.open(tif_path) as src:
        image = src.read().astype(np.float32)
    if image.shape[0] < NUM_BANDS:
        pad = np.zeros((NUM_BANDS - image.shape[0], *image.shape[1:]), dtype=np.float32)
        image = np.concatenate([image, pad], axis=0)
    elif image.shape[0] > NUM_BANDS:
        image = image[:NUM_BANDS]

    nodata = (image.sum(axis=0) == 0)
    image  = np.clip(image, 0.0001, 0.5)
    for b in range(NUM_BANDS):
        image[b] = (image[b] - band_means[b]) / (band_stds[b] + 1e-6)
    image[:, nodata] = 0.0
    image = np.clip(image, -5.0, 5.0)
    image = np.nan_to_num(image, nan=0.0, posinf=5.0, neginf=-5.0)
    return image, nodata


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
@torch.no_grad()
def infer(model, image_np, device, use_tta=False):
    t = torch.from_numpy(image_np).unsqueeze(0).float().to(device)
    if use_tta:
        preds = []
        def _fwd(x):
            return F.softmax(torch.clamp(model(x), -30.0, 30.0), dim=1)
        preds.append(_fwd(t))
        preds.append(_fwd(torch.flip(t,[-1])).flip([-1]))
        preds.append(_fwd(torch.flip(t,[-2])).flip([-2]))
        preds.append(torch.rot90(_fwd(torch.rot90(t,1,[-2,-1])), -1, [-2,-1]))
        preds.append(torch.rot90(_fwd(torch.rot90(t,2,[-2,-1])), -2, [-2,-1]))
        preds.append(torch.rot90(_fwd(torch.rot90(t,3,[-2,-1])), -3, [-2,-1]))
        probs = torch.stack(preds).mean(0)[0].cpu().numpy()
    else:
        logits = torch.clamp(model(t), -30.0, 30.0)
        probs  = F.softmax(logits, dim=1)[0].cpu().numpy()
    return probs.argmax(axis=0), probs


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def patch_metrics(pred_mask, gt_raw, nodata_mask):
    """
    Compute per-class IoU + debris-specific P/R/F1 for one patch.
    gt_raw: raw mask with values 1–15 (1-indexed) or 0 (unlabeled)
    """
    # Remap GT: 1–15 → 0–14 valid, 0 → 255 ignore
    gt = np.where(gt_raw > 0, gt_raw - 1, 255).astype(np.int64)
    gt[nodata_mask] = 255

    valid = gt != 255
    if valid.sum() == 0:
        return None   # No labeled pixels

    pred_v = pred_mask[valid]
    gt_v   = gt[valid]

    # Per-class IoU
    iou_per_cls = {}
    for cls in range(NUM_CLASSES):
        p = pred_v == cls
        g = gt_v   == cls
        inter = (p & g).sum()
        union = (p | g).sum()
        iou_per_cls[cls] = float(inter) / (float(union) + 1e-9) if union > 0 else float("nan")

    # Debris-specific (class 0)
    pred_d = pred_v == 0
    gt_d   = gt_v   == 0
    tp = (pred_d &  gt_d).sum()
    fp = (pred_d & ~gt_d).sum()
    fn = (~pred_d & gt_d).sum()
    gt_debris_n = int(gt_d.sum())

    if gt_debris_n > 0 or fp > 0:
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        iou_d     = tp / (tp + fp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
    else:
        precision = recall = iou_d = f1 = float("nan")

    return {
        "iou_per_class": iou_per_cls,
        "debris_iou":    float(iou_d),
        "debris_f1":     float(f1),
        "debris_prec":   float(precision),
        "debris_recall": float(recall),
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "gt_debris_px":  gt_debris_n,
        "labeled_px":    int(valid.sum()),
    }


# ─────────────────────────────────────────────
# REPORT CHART
# ─────────────────────────────────────────────
def make_report(all_results, output_dir):
    debris_results = [r for r in all_results if not np.isnan(r["debris_iou"])]
    n = len(debris_results)

    if n == 0:
        print("  ⚠️  No patches with GT debris found for charting")
        return

    ious      = [r["debris_iou"]    for r in debris_results]
    f1s       = [r["debris_f1"]     for r in debris_results]
    precs     = [r["debris_prec"]   for r in debris_results]
    recalls   = [r["debris_recall"] for r in debris_results]
    gt_sizes  = [r["gt_debris_px"]  for r in debris_results]
    patches   = [r["patch"]         for r in debris_results]

    # ── Figure ──────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12), facecolor="#0d1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    def ttl(ax, t, c="#58a6ff"):
        ax.set_title(t, color=c, fontweight="bold", fontsize=10, pad=8)

    # 1. IoU distribution
    axes[0].hist(ious, bins=20, color="#3A86FF", edgecolor="#0d1117", alpha=0.85)
    axes[0].axvline(np.nanmean(ious), color="#E63946", lw=2,
                    label=f"Mean={np.nanmean(ious):.3f}")
    axes[0].axvline(np.nanmedian(ious), color="#57CC99", lw=2, ls="--",
                    label=f"Median={np.nanmedian(ious):.3f}")
    axes[0].legend(fontsize=8, labelcolor="white", facecolor="#161b22")
    axes[0].set_xlabel("IoU"); axes[0].set_ylabel("Patches")
    ttl(axes[0], "Marine Debris IoU Distribution")

    # 2. F1 distribution
    axes[1].hist(f1s, bins=20, color="#57CC99", edgecolor="#0d1117", alpha=0.85)
    axes[1].axvline(np.nanmean(f1s), color="#E63946", lw=2,
                    label=f"Mean={np.nanmean(f1s):.3f}")
    axes[1].legend(fontsize=8, labelcolor="white", facecolor="#161b22")
    axes[1].set_xlabel("F1"); axes[1].set_ylabel("Patches")
    ttl(axes[1], "Marine Debris F1 Distribution")

    # 3. Precision vs Recall scatter
    axes[2].scatter(recalls, precs, c=ious, cmap="RdYlGn",
                    s=40, alpha=0.75, edgecolors="none", vmin=0, vmax=1)
    axes[2].plot([0, 1], [0, 1], "w--", alpha=0.3)
    axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision")
    axes[2].set_xlim(0, 1.05); axes[2].set_ylim(0, 1.05)
    ttl(axes[2], "Precision vs Recall  (colour=IoU)")

    # 4. Per-class mean IoU bar chart
    all_class_ious = defaultdict(list)
    for r in all_results:
        for cls, iou in r["iou_per_class"].items():
            if not np.isnan(iou):
                all_class_ious[cls].append(iou)

    mean_cls_iou = {cls: np.mean(v) for cls, v in all_class_ious.items() if v}
    sorted_cls   = sorted(mean_cls_iou, key=mean_cls_iou.get, reverse=True)
    bar_labels   = [CLASS_MAP.get(c, str(c)) for c in sorted_cls]
    bar_vals     = [mean_cls_iou[c] for c in sorted_cls]
    bar_colors   = [CLASS_COLORS[c] for c in sorted_cls]

    bars = axes[3].barh(range(len(bar_labels)), bar_vals,
                        color=bar_colors, edgecolor="#0d1117", height=0.7)
    axes[3].set_yticks(range(len(bar_labels)))
    axes[3].set_yticklabels(bar_labels, fontsize=8)
    axes[3].set_xlabel("Mean IoU")
    axes[3].set_xlim(0, 1.05)
    axes[3].axvline(0.5, color="white", lw=0.5, alpha=0.4)
    for bar, val in zip(bars, bar_vals):
        axes[3].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                     f"{val:.3f}", va="center", fontsize=7, color="white")
    ttl(axes[3], "Mean IoU per Class")

    # 5. IoU vs GT debris pixel count
    axes[4].scatter(gt_sizes, ious, c="#E9C46A", s=30, alpha=0.6, edgecolors="none")
    axes[4].set_xlabel("GT Debris Pixels"); axes[4].set_ylabel("Debris IoU")
    axes[4].set_xscale("log")
    ttl(axes[4], "IoU vs GT Debris Patch Size")

    # 6. Summary stats text box
    metrics_text = [
        ("Patches evaluated",    f"{len(all_results)}"),
        ("Patches with debris",  f"{n}"),
        ("Mean Debris IoU",      f"{np.nanmean(ious):.4f}"),
        ("Median Debris IoU",    f"{np.nanmedian(ious):.4f}"),
        ("Mean Debris F1",       f"{np.nanmean(f1s):.4f}"),
        ("Mean Precision",       f"{np.nanmean(precs):.4f}"),
        ("Mean Recall",          f"{np.nanmean(recalls):.4f}"),
        ("IoU > 0.5  patches",   f"{sum(i > 0.5 for i in ious)} / {n}"),
        ("IoU > 0.7  patches",   f"{sum(i > 0.7 for i in ious)} / {n}"),
        ("IoU > 0.9  patches",   f"{sum(i > 0.9 for i in ious)} / {n}"),
    ]
    axes[5].axis("off")
    y = 0.95
    axes[5].text(0.05, y, "📊 Evaluation Summary", color="#58a6ff",
                 fontsize=11, fontweight="bold", transform=axes[5].transAxes)
    y -= 0.10
    for label, val in metrics_text:
        axes[5].text(0.05, y, label, color="#8b949e", fontsize=9,
                     transform=axes[5].transAxes)
        axes[5].text(0.72, y, val, color="white", fontsize=9, fontweight="bold",
                     transform=axes[5].transAxes)
        y -= 0.085

    plt.suptitle("MARIDA Model Evaluation Report", color="white",
                 fontsize=14, fontweight="bold", y=0.98)

    out = output_dir / "evaluation_report.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"\n  📈 Report saved → {out}")
    return out


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split",    type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--output",   type=str, default="evaluation")
    parser.add_argument("--tta",      action="store_true",
                        help="Use test-time augmentation (8 variants, slower but more accurate)")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n🖥  Device : {device}")
    print(f"📁 Data   : {data_dir}")
    print(f"📊 Split  : {args.split}\n")

    model, band_means, band_stds = load_model(args.model, device)
    print(f"  TTA enabled : {args.tta}")

    # Load split
    split_file = data_dir / "splits" / f"{args.split}_X.txt"
    patch_names = split_file.read_text().strip().splitlines()

    # Build lookup
    all_tifs    = list(data_dir.rglob("*.tif"))
    img_lookup  = {f.stem: f for f in all_tifs
                   if "_cl" not in f.stem and "_conf" not in f.stem}
    mask_lookup = {f.stem.replace("_cl", ""): f for f in all_tifs if "_cl" in f.stem}

    print(f"Processing {len(patch_names)} patches...\n")
    print(f"{'Patch':<35} {'Labeled':>8} {'DebrisIoU':>10} {'F1':>8} {'P':>8} {'R':>8}")
    print("─" * 80)

    all_results = []
    skipped = 0

    for name in patch_names:
        name      = name.strip()
        disk_name = f"S2_{name}" if not name.upper().startswith("S2_") else name

        if disk_name not in img_lookup or disk_name not in mask_lookup:
            skipped += 1
            continue

        img_path = img_lookup[disk_name]
        msk_path = mask_lookup[disk_name]

        try:
            image_np, nodata = preprocess(img_path, band_means, band_stds)
            pred_mask, probs = infer(model, image_np, device, use_tta=args.tta)
            pred_mask[nodata] = 6  # nodata → Marine Water

            with rasterio.open(msk_path) as src:
                gt_raw = src.read(1).astype(np.int32)

            result = patch_metrics(pred_mask, gt_raw, nodata)
            if result is None:
                skipped += 1
                continue

            result["patch"] = disk_name
            all_results.append(result)

            iou_str = f"{result['debris_iou']:.4f}" if not np.isnan(result['debris_iou']) else "  N/A  "
            f1_str  = f"{result['debris_f1']:.4f}"  if not np.isnan(result['debris_f1'])  else "  N/A  "
            p_str   = f"{result['debris_prec']:.4f}" if not np.isnan(result['debris_prec']) else "  N/A  "
            r_str   = f"{result['debris_recall']:.4f}" if not np.isnan(result['debris_recall']) else "  N/A  "
            print(f"  {disk_name:<33} {result['labeled_px']:>8,} {iou_str:>10} "
                  f"{f1_str:>8} {p_str:>8} {r_str:>8}")

        except Exception as e:
            print(f"  ⚠️  {disk_name}: {e}")
            skipped += 1

    print(f"\n  Processed: {len(all_results)} | Skipped: {skipped}")

    # ── Aggregate metrics ─────────────────────────────────
    debris_results = [r for r in all_results if not np.isnan(r["debris_iou"])]
    n_debris = len(debris_results)

    print(f"\n{'='*60}")
    print(f"  AGGREGATE METRICS  ({args.split.upper()} SET)")
    print(f"{'='*60}")
    print(f"  Total patches           : {len(all_results)}")
    print(f"  Patches with GT debris  : {n_debris}")

    if n_debris > 0:
        ious    = [r["debris_iou"]    for r in debris_results]
        f1s     = [r["debris_f1"]     for r in debris_results]
        precs   = [r["debris_prec"]   for r in debris_results]
        recalls = [r["debris_recall"] for r in debris_results]

        print(f"\n  Marine Debris Metrics:")
        print(f"  {'Mean IoU':<25}: {np.nanmean(ious):.4f}")
        print(f"  {'Median IoU':<25}: {np.nanmedian(ious):.4f}")
        print(f"  {'Mean F1':<25}: {np.nanmean(f1s):.4f}")
        print(f"  {'Mean Precision':<25}: {np.nanmean(precs):.4f}")
        print(f"  {'Mean Recall':<25}: {np.nanmean(recalls):.4f}")
        print(f"  {'IoU > 0.5 patches':<25}: {sum(i>0.5 for i in ious)}/{n_debris}")
        print(f"  {'IoU > 0.7 patches':<25}: {sum(i>0.7 for i in ious)}/{n_debris}")
        print(f"  {'IoU > 0.9 patches':<25}: {sum(i>0.9 for i in ious)}/{n_debris}")

    # ── Per-class IoU ─────────────────────────────────────
    print(f"\n  Per-class Mean IoU ({args.split}):")
    print(f"  {'Class':<28} {'Mean IoU':>10}  {'Patches':>8}")
    print("  " + "─" * 50)
    class_ious = defaultdict(list)
    for r in all_results:
        for cls, iou in r["iou_per_class"].items():
            if not np.isnan(iou):
                class_ious[cls].append(iou)
    for cls in range(NUM_CLASSES):
        vals = class_ious.get(cls, [])
        miou = np.mean(vals) if vals else float("nan")
        marker = " ★" if cls == 0 else ""
        print(f"  {CLASS_MAP[cls]:<28} {miou:>10.4f}  {len(vals):>8}{marker}")

    # ── Save results ──────────────────────────────────────
    out_json = output_dir / f"{args.split}_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  💾 Per-patch results → {out_json}")

    if HAS_PANDAS:
        import pandas as pd
        rows = []
        for r in all_results:
            row = {
                "patch":         r["patch"],
                "labeled_px":    r["labeled_px"],
                "gt_debris_px":  r["gt_debris_px"],
                "debris_iou":    r["debris_iou"],
                "debris_f1":     r["debris_f1"],
                "debris_prec":   r["debris_prec"],
                "debris_recall": r["debris_recall"],
                "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
            }
            for cls in range(NUM_CLASSES):
                row[f"iou_{CLASS_MAP[cls].replace(' ','_')}"] = \
                    r["iou_per_class"].get(cls, float("nan"))
            rows.append(row)
        df = pd.DataFrame(rows)
        csv_path = output_dir / f"{args.split}_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"  📋 CSV report       → {csv_path}")

    # ── Report chart ──────────────────────────────────────
    make_report(all_results, output_dir)
    print(f"\n✅ Evaluation complete — results in: {output_dir}/\n")


if __name__ == "__main__":
    main()
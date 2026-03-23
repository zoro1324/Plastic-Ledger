"""
Multi-Model Comparative Evaluation
====================================
Evaluates three MARIDA models on the official test set and prints a
side-by-side comparison table:
  - SegFormer (our custom-trained model)
  - U-Net     (official MARIDA pretrained baseline)
  - ResNet    (official MARIDA multi-label baseline — image-level F1)

Usage:
  python evaluate_models.py
"""

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rasterio
from torch.utils.data import DataLoader

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Shared settings
# ─────────────────────────────────────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR     = r"d:\Plastic-Ledger\U-net-models\dataset\MARIDA"
PATCHES_DIR  = os.path.join(DATA_DIR, "patches")
TEST_SPLIT   = os.path.join(DATA_DIR, "splits", "test_X.txt")
NUM_CLASSES  = 15          # foreground classes (background mapped to 255)
IGNORE_IDX   = 255

CLASS_NAMES = [
    "Marine Debris", "Dense Sargassum", "Sparse Sargassum",
    "Natural Organic Material", "Ship", "Clouds", "Marine Water",
    "Sediment-Laden Water", "Foam", "Turbid Water", "Shallow Water",
    "Waves", "Cloud Shadows", "Wakes", "Mixed Water",
]

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Dataset (identical to SegFormer training dataset.py)
# ─────────────────────────────────────────────────────────────────────────────
class MaridaTestDataset(torch.utils.data.Dataset):
    def __init__(self, splits_file, patches_dir):
        self.patches_dir = patches_dir
        with open(splits_file) as f:
            self.names = [l.strip() for l in f if l.strip()]
        first = self._img_path(self.names[0])
        with rasterio.open(first) as s:
            self.in_channels = s.count

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
                torch.tensor(mask, dtype=torch.long))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Metrics helper
# ─────────────────────────────────────────────────────────────────────────────
class SegMetrics:
    def __init__(self):
        self.inter = np.zeros(NUM_CLASSES, np.float64)
        self.union = np.zeros(NUM_CLASSES, np.float64)
        self.correct = self.total = 0

    def update(self, pred, target):
        p = pred.cpu().numpy().flatten()
        t = target.cpu().numpy().flatten()
        v = t != IGNORE_IDX
        p, t = p[v], t[v]
        self.correct += (p == t).sum()
        self.total   += v.sum()
        for c in range(NUM_CLASSES):
            pc = p == c; tc = t == c
            self.inter[c] += (pc & tc).sum()
            self.union[c] += (pc | tc).sum()

    def iou(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.union > 0, self.inter / self.union, np.nan)

    def miou(self): return float(np.nanmean(self.iou()))
    def pix_acc(self): return float(self.correct / (self.total + 1e-9))


# ─────────────────────────────────────────────────────────────────────────────
# 3a.  U-Net — exact structural match to the official MARIDA checkpoint keys
#      inc.{0,1,3,4}.*  /  down{N}.maxpool_conv.{1,2,4,5}.*  /  up{N}.conv.{0,1,3,4}.*
# ─────────────────────────────────────────────────────────────────────────────
def _double_conv_flat(in_ch, out_ch):
    """Returns nn.Sequential with indices 0,1,2,3,4 matching MARIDA flat layout."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),   # .0
        nn.BatchNorm2d(out_ch),                               # .1
        nn.ReLU(inplace=True),                               # .2  (no params — skipped in sd)
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),  # .3
        nn.BatchNorm2d(out_ch),                              # .4
        nn.ReLU(inplace=True),                               # .5  (no params)
    )


class UNet(nn.Module):
    """Exact mirror of the MARIDA official U-Net state dict layout."""
    def __init__(self, in_channels=11, num_classes=11, hidden=16):
        super().__init__()
        h = hidden
        # inc  → flat double conv (keys: inc.0.* inc.1.* inc.3.* inc.4.*)
        self.inc = _double_conv_flat(in_channels, h)

        # down blocks — maxpool_conv is a flat Sequential:
        #   [0]=MaxPool2d, [1]=Conv, [2]=BN, [3]=ReLU, [4]=Conv, [5]=BN, [6]=ReLU
        def _down(ic, oc):
            return nn.Sequential(
                nn.MaxPool2d(2),                                   # .0
                nn.Conv2d(ic, oc, 3, padding=1, bias=True),        # .1
                nn.BatchNorm2d(oc),                                # .2
                nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, padding=1, bias=True),        # .4
                nn.BatchNorm2d(oc),                                # .5
                nn.ReLU(inplace=True),
            )

        class Down(nn.Module):
            def __init__(self, ic, oc):
                super().__init__()
                self.maxpool_conv = _down(ic, oc)
            def forward(self, x): return self.maxpool_conv(x)

        self.down1 = Down(h,    h*2)   # 16 → 32
        self.down2 = Down(h*2,  h*4)   # 32 → 64
        self.down3 = Down(h*4,  h*8)   # 64 → 128
        self.down4 = Down(h*8,  h*8)   # 128 → 128  (MARIDA caps at 128, no doubling)

        # up blocks — conv is a flat sequential mirroring _double_conv_flat
        class Up(nn.Module):
            def __init__(self, ic, oc):
                super().__init__()
                self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                self.conv = _double_conv_flat(ic, oc)
            def forward(self, x1, x2):
                x1 = self.up(x1)
                return self.conv(torch.cat([x2, x1], dim=1))

        self.up1 = Up(128+128, 64)   # x5(128) cat x4(128) → 256 in, 64 out
        self.up2 = Up(64 +64,  32)   # x(64)  cat x3(64)  → 128 in, 32 out
        self.up3 = Up(32 +32,  16)   # x(32)  cat x2(32)  →  64 in, 16 out
        self.up4 = Up(16 +16,  16)   # x(16)  cat x1(16)  →  32 in, 16 out
        self.outc = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3b.  ResNet — exact MARIDA multi-label baseline key structure
#      encoder.0 = conv1, encoder.1 = bn1, encoder.4/5/6/7 = layer1-4, fc = FC
# ─────────────────────────────────────────────────────────────────────────────
class MaridaResNet(nn.Module):
    """Wraps torchvision ResNet-50 with the MARIDA key naming scheme."""
    def __init__(self, in_channels=11, num_classes=11):
        super().__init__()
        import torchvision.models as tvm
        base = tvm.resnet50(weights=None)
        # Adapt first conv for N bands
        base.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        # Map to MARIDA keys: encoder.0=conv1, encoder.1=bn1, encoder.4=layer1...
        self.encoder = nn.Sequential(
            base.conv1,       # encoder.0
            base.bn1,         # encoder.1
            base.relu,        # encoder.2  (no params)
            base.maxpool,     # encoder.3  (no params)
            base.layer1,      # encoder.4
            base.layer2,      # encoder.5
            base.layer3,      # encoder.6
            base.layer4,      # encoder.7
            base.avgpool,     # encoder.8  (no params)
        )
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        feat = self.encoder(x).flatten(1)
        return self.fc(feat)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SegFormer  (HuggingFace wrapper — same as train.py)
# ─────────────────────────────────────────────────────────────────────────────
def build_segformer(in_channels, num_classes):
    from transformers import SegformerForSemanticSegmentation, SegformerConfig
    cfg = SegformerConfig(
        num_labels=num_classes,
        num_channels=in_channels,
        depths=[2, 2, 2, 2],
        hidden_sizes=[32, 64, 160, 256],
        decoder_hidden_size=256,
    )
    model = SegformerForSemanticSegmentation(cfg)
    # Overwrite the first projection to accept 11 channels
    old = model.segformer.encoder.patch_embeddings[0].proj
    new = nn.Conv2d(in_channels, old.out_channels,
                    kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding)
    nn.init.kaiming_normal_(new.weight)
    model.segformer.encoder.patch_embeddings[0].proj = new
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Evaluation loops
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_segmentation(model, loader, resize_logits=True):
    """For pixel-wise models: SegFormer & U-Net."""
    model.eval()
    m = SegMetrics()
    for imgs, masks in loader:
        imgs  = imgs.to(DEVICE)
        masks = masks.to(DEVICE)
        out = model(imgs)
        if hasattr(out, "logits"):
            logits = out.logits   # SegFormer
        else:
            logits = out          # U-Net
        if resize_logits and logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:],
                                   mode="bilinear", align_corners=False)
        preds = logits.argmax(1)
        m.update(preds, masks)
    return m


@torch.no_grad()
def eval_resnet(model, loader):
    """For patch-level ResNet: assign each patch's top predicted class to all
    valid pixels in that mask, then compute pixel accuracy over the test set."""
    model.eval()
    correct = total = 0
    for imgs, masks in loader:
        imgs  = imgs.to(DEVICE)
        masks = masks.to(DEVICE)
        logits = model(imgs)                    # (B, 15)
        pred_labels = logits.argmax(1)          # (B,)
        # Expand patch label to pixel level
        B, H, W = masks.shape
        pred_maps = pred_labels[:, None, None].expand(B, H, W)
        valid = masks != IGNORE_IDX
        correct += (pred_maps[valid] == masks[valid]).sum().item()
        total   += valid.sum().item()
    pix_acc = correct / (total + 1e-9)
    return pix_acc


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ds = MaridaTestDataset(TEST_SPLIT, PATCHES_DIR)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    print(f"Test set: {len(ds)} patches | Device: {DEVICE}\n")

    results = {}

    # ── SegFormer ──────────────────────────────────────────────────────────
    print("Loading SegFormer …")
    sg = build_segformer(ds.in_channels, NUM_CLASSES).to(DEVICE)
    sg.load_state_dict(
        torch.load(r"best_model_SegTransformer.pth",
                   map_location=DEVICE, weights_only=False),
        strict=False   # handles any minor key mismatches
    )
    sg_m = eval_segmentation(sg, loader)
    results["SegFormer"] = {"miou": sg_m.miou(), "pix_acc": sg_m.pix_acc(),
                            "iou_per_class": sg_m.iou()}
    del sg
    torch.cuda.empty_cache()

    # ── U-Net ──────────────────────────────────────────────────────────────
    print("Loading U-Net …")
    # Infer hidden_channels from state dict (outc weight shape)
    unet_sd = torch.load(r"best_model_UNet.pth",
                         map_location=DEVICE, weights_only=False)
    unet_classes = unet_sd["outc.weight"].shape[0]  # auto-detect (e.g. 11 or 15)
    hidden       = unet_sd["outc.weight"].shape[1]
    print(f"  U-Net detected: hidden={hidden}, num_classes={unet_classes}")
    unet = UNet(in_channels=ds.in_channels, num_classes=unet_classes, hidden=hidden).to(DEVICE)
    unet.load_state_dict(unet_sd, strict=True)
    un_m = eval_segmentation(unet, loader, resize_logits=False)
    results["U-Net"] = {"miou": un_m.miou(), "pix_acc": un_m.pix_acc(),
                        "iou_per_class": un_m.iou()}
    del unet, unet_sd
    torch.cuda.empty_cache()

    # ── ResNet ──────────────────────────────────────────────────────────────
    print("Loading ResNet-50 …")
    rn_sd = torch.load(r"best_model_resnet.pth",
                       map_location=DEVICE, weights_only=False)
    fc_out = rn_sd["fc.weight"].shape[0]
    rnet = MaridaResNet(in_channels=ds.in_channels, num_classes=fc_out).to(DEVICE)
    rnet.load_state_dict(rn_sd, strict=True)
    rn_acc = eval_resnet(rnet, loader)
    results["ResNet-50"] = {"pix_acc": rn_acc, "miou": None, "iou_per_class": None}
    del rnet, rn_sd
    torch.cuda.empty_cache()

    # ── Summary Table ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("          MARIDA MODEL COMPARISON — TEST SET RESULTS")
    print("="*60)
    print(f"{'Model':<14} {'Pixel Acc':>10} {'Mean IoU':>10}  Notes")
    print("-"*60)
    for name, d in results.items():
        acc_str  = f"{d['pix_acc']*100:.2f}%"
        miou_str = f"{d['miou']:.4f}" if d["miou"] is not None else "N/A (patch-level)"
        note = "(patch-level classifier)" if d["miou"] is None else ""
        print(f"{name:<14} {acc_str:>10} {miou_str:>10}  {note}")
    print("="*60)

    # ── Per-class IoU ──────────────────────────────────────────────────────
    seg_models = {k: v for k, v in results.items() if v["iou_per_class"] is not None}
    if seg_models:
        print(f"\n{'Class':<28}", end="")
        for name in seg_models: print(f"  {name:>10}", end="")
        print()
        print("-" * (28 + 13 * len(seg_models)))
        for i, cls in enumerate(CLASS_NAMES):
            print(f"{cls:<28}", end="")
            for d in seg_models.values():
                v = d["iou_per_class"][i]
                print(f"  {v:>10.4f}" if not np.isnan(v) else f"  {'N/A':>10}", end="")
            print()

    # ── Verdict ────────────────────────────────────────────────────────────
    seg_miou = {k: v["miou"] for k, v in results.items() if v["miou"] is not None}
    winner = max(seg_miou, key=seg_miou.get)
    print(f"\n🏆  Best segmentation model: {winner}  (mIoU = {seg_miou[winner]:.4f})\n")


if __name__ == "__main__":
    main()

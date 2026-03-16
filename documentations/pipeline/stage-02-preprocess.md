# Stage 2: Preprocessing

**File:** `pipeline/02_preprocess.py`

---

## Overview

Stage 2 transforms raw Sentinel-2 per-band GeoTIFFs (from Stage 1) into **model-ready 256×256 overlapping patches**. The process includes:

1. Loading and reordering 8 available bands into the 11-band model input format (zero-padding missing bands)
2. Z-score normalization using MARIDA dataset statistics
3. Tiling the full scene into overlapping patches
4. Saving patches as `.npy` / `.npz` files with a geo-index

---

## Input

| Parameter | Type | Description |
|-----------|------|-------------|
| `scene_dir` | `Path` | Raw scene directory from Stage 1 containing `<BAND>.tif` files |
| `output_dir` | `Path` | Root output directory (default: `data/processed`) |
| `config` | `dict` | Optional config with `patch_size`, `overlap`, `patch_storage`, `patch_dtype` |

---

## Output

```
data/processed/<SCENE_ID>/
├── patches/
│   ├── patch_0000.npy   (or .npz)
│   ├── patch_0001.npy
│   └── ...
├── patch_index.json     # Geo-coordinates and pixel offsets for every patch
├── nodata_mask.npy      # Boolean mask (H×W) of all-zero pixels
└── scene_meta.json      # Original shape, CRS, transform, patch settings
```

**Returns:** `(patches_dir: Path, patch_index: Dict)`

---

## Band Mapping

The model expects **11 bands** in this exact order:

| Position | Band | Available? |
|----------|------|-----------|
| 0 | B01 (Coastal aerosol) | ❌ Zero-padded |
| 1 | B02 (Blue) | ✅ |
| 2 | B03 (Green) | ✅ |
| 3 | B04 (Red) | ✅ |
| 4 | B05 (Red Edge 1) | ✅ |
| 5 | B06 (Red Edge 2) | ❌ Zero-padded |
| 6 | B07 (Red Edge 3) | ❌ Zero-padded |
| 7 | B08 (NIR broad) | ✅ |
| 8 | B8A (NIR narrow) | ✅ |
| 9 | B11 (SWIR 1) | ✅ |
| 10 | B12 (SWIR 2) | ✅ |

**Mapping (8 available bands → 11 model positions):**
```python
mapping = {0: 1, 1: 2, 2: 3, 3: 4, 4: 7, 5: 8, 6: 9, 7: 10}
# Available index → model position
# B02(0)→1, B03(1)→2, B04(2)→3, B05(3)→4,
# B08(4)→7, B8A(5)→8, B11(6)→9, B12(7)→10
# Positions 0 (B01), 5 (B06), 6 (B07) are filled with zeros
```

---

## Normalization Formula

### Step 1 — Clip raw reflectance
```
pixel_value = clip(pixel_value, 0.0001, 0.5)
```
This prevents extreme outliers (sensor saturation, deep shadows).

### Step 2 — Z-score per band
```
normalized[b] = (raw[b] - mean[b]) / (std[b] + 1e-6)
```

**Per-band MARIDA statistics (11 bands):**

| Position | Band | Mean | Std |
|----------|------|------|-----|
| 0 | B01 | 0.057 | 0.010 |
| 1 | B02 | 0.054 | 0.010 |
| 2 | B03 | 0.046 | 0.013 |
| 3 | B04 | 0.036 | 0.010 |
| 4 | B05 | 0.033 | 0.012 |
| 5 | B06 | 0.041 | 0.020 |
| 6 | B07 | 0.049 | 0.030 |
| 7 | B08 | 0.043 | 0.020 |
| 8 | B8A | 0.050 | 0.030 |
| 9 | B11 | 0.031 | 0.020 |
| 10 | B12 | 0.019 | 0.013 |

### Step 3 — Safety clip
```
normalized = clip(normalized, -5.0, 5.0)
```
Handles any remaining NaN/Inf → set to 0.

### Step 4 — Nodata masking
Pixels where **all bands sum to zero** are marked as nodata and reset to 0 after normalization:
```python
nodata_mask = (image.sum(axis=0) == 0)
image[:, nodata_mask] = 0.0
```

---

## Tiling

The scene is divided into overlapping 256×256 patches:

### Parameters (from `config.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patch_size` | 256 | Patch width and height in pixels |
| `overlap` | 32 | Overlap between adjacent patches in pixels |
| `patch_storage` | `npz` | File format: `npy` (raw) or `npz` (compressed) |
| `patch_dtype` | `float16` | Data type: `float32` or `float16` (halves disk usage) |

### Stride Calculation

```
stride = patch_size - overlap = 256 - 32 = 224 pixels
```

### Grid Position

For a scene of size `H × W`:
```
n_rows = ceil(H / stride)
n_cols = ceil(W / stride)
total_patches = n_rows × n_cols
```

Edge patches are **zero-padded** to fill the full 256×256 size.

---

## Patch Index Structure

Each entry in `patch_index.json`:
```json
{
  "patch_0000": {
    "row": 0,
    "col": 0,
    "row_start": 0,
    "col_start": 0,
    "actual_h": 256,
    "actual_w": 256,
    "patch_file": "patch_0000.npz",
    "geo_transform": [10.0, 0.0, 796260.0, 0.0, -10.0, 1115590.0],
    "crs": "EPSG:32644",
    "nodata_mask_path": "data/processed/<SCENE_ID>/nodata_mask.npy"
  }
}
```

The `geo_transform` is an Affine transform `[a, b, c, d, e, f]` per the GDAL convention: `x = c + col*a`, `y = f + row*e`.

---

## Multi-resolution Handling

Sentinel-2 bands have different ground sampling distances:
- **10 m:** B02, B03, B04, B08
- **20 m:** B05, B8A, B11, B12

When individual 20 m bands are smaller than the 10 m reference grid, `scipy.ndimage.zoom` resamples them to match:
```python
zoom_factors = (ref_h / band_h, ref_w / band_w)
arr = zoom(arr, zoom_factors, order=1)  # bilinear
```

---

## Caching

If `patch_index.json` already exists in the output directory, the entire stage is skipped — patches are re-used from the previous run.

---

## Processing Steps

```
1. Load per-band TIF files from scene_dir
       ↓
2. Resample any 20 m bands to match 10 m reference grid
       ↓
3. Assemble 8-band array → pad to 11-band model order
       ↓
4. Detect nodata pixels (all-band sum == 0)
       ↓
5. Clip [0.0001, 0.5] → Z-score normalize → clip [-5, 5]
       ↓
6. Save nodata_mask.npy
       ↓
7. Tile into 256×256 patches with 32-pixel overlap
       ↓
8. Save each patch as patch_NNNN.npz (or .npy)
       ↓
9. Write patch_index.json and scene_meta.json
```

---

## CLI Usage

```bash
python -m pipeline.02_preprocess \
    --scene_dir  data/raw/S2A_MSIL2A_... \
    --output_dir data/processed \
    --config     config/config.yaml
```

---

## Config Keys Used

```yaml
preprocessing:
  patch_size:    256
  overlap:        32
  patch_storage: npz     # npy or npz
  patch_dtype:   float16 # float32 or float16
  band_means: [0.057, 0.054, 0.046, 0.036, 0.033, 0.041, 0.049, 0.043, 0.050, 0.031, 0.019]
  band_stds:  [0.010, 0.010, 0.013, 0.010, 0.012, 0.020, 0.030, 0.020, 0.030, 0.020, 0.013]
```

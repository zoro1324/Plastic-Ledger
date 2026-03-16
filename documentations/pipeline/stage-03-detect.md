# Stage 3: Marine Debris Detection

**File:** `pipeline/03_detect.py`

---

## Overview

Stage 3 is the core machine-learning inference stage. It loads a trained **U-Net segmentation model** and runs it over every preprocessed 256×256 patch from Stage 2. Predictions from all patches are stitched back into a full-scene probability map, then thresholded and clustered into **georeferenced debris detection objects** exported as GeoJSON.

---

## Input

| Parameter | Type | Description |
|-----------|------|-------------|
| `scene_id` | `str` | Identifier of the scene being processed |
| `patches_dir` | `Path` | Directory containing `patch_NNNN.npy/.npz` files (from Stage 2) |
| `model_path` | `Path` | Path to the trained `.pth` model checkpoint |
| `output_dir` | `Path` | Root directory for detection outputs (default: `data/detections`) |
| `config` | `dict` | Optional config with threshold, TTA flag, min area settings |

---

## Output

```
data/detections/<SCENE_ID>/
├── detections.geojson   # Georeferenced debris cluster polygons
├── debris_mask.tif      # Binary mask (0/1): GeoTIFF, uint8
├── debris_prob.tif      # Per-pixel debris probability: GeoTIFF, float32
└── class_mask.tif       # Argmax class assignment per pixel: GeoTIFF, uint8
```

**Returns:** `Path` to `detections.geojson`

---

## Model Architecture

### U-Net with ResNet-34 Encoder

```
Architecture: segmentation_models_pytorch.Unet
  encoder_name:    resnet34
  encoder_weights: None  (loaded from checkpoint)
  in_channels:     11    (11-band Sentinel-2 input)
  classes:         15    (15 marine surface classes)
  activation:      None  (raw logits, softmax applied in inference)
```

### Checkpoint Format

```python
{
  "model_state": OrderedDict(...),   # PyTorch state dict
  "encoder":     "resnet34",         # Encoder name
  "num_bands":   11,
  "num_classes": 15,
  "epoch":       <int>,              # Training epoch
}
```

---

## Class Map (15 Classes)

| Index | Class Name |
|-------|-----------|
| **0** | **Marine Debris** ← target class |
| 1 | Dense Sargassum |
| 2 | Sparse Sargassum |
| 3 | Natural Organic Material |
| 4 | Ship |
| 5 | Clouds |
| 6 | Marine Water |
| 7 | Sediment-Laden Water |
| 8 | Foam |
| 9 | Turbid Water |
| 10 | Shallow Water |
| 11 | Waves |
| 12 | Cloud Shadows |
| 13 | Wakes |
| 14 | Mixed Water |

---

## Inference Pipeline

### Step 1 — Patch Loading

Each patch is loaded from `.npy` or `.npz` format as a `(11, 256, 256)` float32/float16 array.

### Step 2 — Test-Time Augmentation (TTA)

6 augmentation variants are applied to each patch:

| Augment | Forward Transform | Reverse Transform |
|---------|-------------------|-------------------|
| `original` | identity | identity |
| `hflip` | flip axis=2 (horizontal) | flip axis=2 |
| `vflip` | flip axis=1 (vertical) | flip axis=1 |
| `rot90` | rot90 k=1 axes=(1,2) | rot90 k=-1 |
| `rot180` | rot90 k=2 axes=(1,2) | rot90 k=-2 |
| `rot270` | rot90 k=3 axes=(1,2) | rot90 k=-3 |

**TTA formula:**
```
prob_map = (1/6) × Σ reverse_aug(softmax(model(aug(patch))))
```

The averaged probability map `(15, 256, 256)` reduces prediction variance caused by orientation artefacts.

### Step 3 — Softmax

```python
probs = F.softmax(logits, dim=1)  # (1, 15, H, W) → (15, H, W)
```
Logits are clamped to `[-30, 30]` before softmax to prevent overflow.

---

## Patch Stitching

The per-patch predictions `(15, 256, 256)` are assembled back into the full-scene probability map `(15, H, W)` using average blending in overlap regions:

```
prob_sum[:, rs:rs+ah, cs:cs+aw] += pred[:, :ah, :aw]
count[rs:rs+ah, cs:cs+aw]       += 1

final_prob = prob_sum / max(count, 1)
```

Where `rs`, `cs` are the patch's `row_start`, `col_start` pixel offsets (from `patch_index.json`).

---

## Debris Masking

Two conditions must both be true for a pixel to be flagged as debris:

```python
debris_prob  = full_probs[0]          # class-0 probability
class_mask   = full_probs.argmax(0)   # predicted class per pixel

debris_mask = (debris_prob > threshold) & (class_mask == 0)
```

| Parameter | Default (config) | Description |
|-----------|-----------------|-------------|
| `debris_threshold` | `0.1` | Minimum debris class probability |

The dual condition prevents high-probability pixels that are *not* the argmax class (i.e., another class is more likely) from being flagged.

---

## Cluster Extraction

Connected components of the debris mask are extracted and converted to georeferenced polygon features:

### Step 1 — Connected Components
```python
labeled, num_features = scipy.ndimage.label(debris_mask.astype(np.int32))
```

### Step 2 — Area Filtering
```
pixel_area_m2 = abs(transform.a × transform.e)
min_pixels    = ceil(min_cluster_area_m2 / pixel_area_m2)
```

Clusters with fewer pixels than `min_pixels` are discarded.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_cluster_area_m2` | `100` | Minimum 4 pixels at 10 m resolution |

### Step 3 — Polygon Conversion

Each cluster mask is converted to Shapely polygons via `rasterio.features` (rasterio's vectorization). All polygons for a cluster are merged with `unary_union`.

### Step 4 — Centroid Re-projection

If the scene CRS is not EPSG:4326, centroids are re-projected for longitude/latitude reporting:
```python
project = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
centroid_lonlat = shapely.ops.transform(project, centroid)
```

---

## Output GeoJSON Schema

Each feature in `detections.geojson`:
```json
{
  "type": "Feature",
  "geometry": { "type": "Polygon", "coordinates": [[...]] },
  "properties": {
    "cluster_id":       0,
    "area_m2":          2500.0,
    "mean_confidence":  0.72,
    "centroid_lon":     81.234,
    "centroid_lat":     8.567,
    "detection_date":   "2024-01-15T10:23:00"
  }
}
```

---

## Processing Steps

```
1. Load model checkpoint → U-Net (ResNet-34 encoder) → eval mode
       ↓
2. Load patch_index.json and scene_meta.json
       ↓
3. For each patch:
     a. Load patch array (.npy / .npz)
     b. Apply 6 TTA augmentations
     c. Run model forward pass for each
     d. Reverse augmentation on output probabilities
     e. Average probabilities across 6 variants
       ↓
4. Stitch all patches into full-scene (15, H, W) probability map
       ↓
5. Extract debris_prob[0] and class_mask = argmax
       ↓
6. Apply dual-condition debris mask
       ↓
7. Save debris_mask.tif, debris_prob.tif, class_mask.tif
       ↓
8. Run scipy connected-components labelling
       ↓
9. Filter clusters by min area
       ↓
10. Convert clusters → Shapely polygons → reproject centroids
       ↓
11. Save detections.geojson (EPSG:4326)
```

---

## Performance Notes

- Inference runs on GPU if available (`torch.cuda.is_available()`), otherwise CPU
- TTA can be disabled via `config.model.tta: false` for faster (but lower-quality) results
- With TTA enabled, each patch requires 6 forward passes

---

## Caching

If `detections.geojson` already exists in the output directory, the entire stage is skipped.

---

## CLI Usage

```bash
python -m pipeline.03_detect \
    --scene_id    S2A_MSIL2A_20240115... \
    --patches_dir data/processed/S2A.../patches \
    --model_path  models/runs/marida_v1/best_model.pth \
    --output_dir  data/detections \
    --config      config/config.yaml
```

---

## Config Keys Used

```yaml
model:
  debris_threshold: 0.1   # Minimum probability to flag a pixel as debris
  tta: true               # Enable Test-Time Augmentation

detection:
  min_cluster_area_m2: 100  # Filter tiny clusters (< 4 pixels at 10 m)
  debris_class_index:  0    # Index of "Marine Debris" in the 15-class output
```

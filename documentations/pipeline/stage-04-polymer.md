# Stage 4: Polymer Type Classification

**File:** `pipeline/04_polymer.py`

---

## Overview

Stage 4 analyses the spectral signature of each detected debris cluster and classifies it by **probable polymer type** using a rule-based spectral decision tree. It computes four spectral indices from the Sentinel-2 reflectance data and maps them to polymer categories, also flagging organic-matter false positives.

---

## Input

| Parameter | Type | Description |
|-----------|------|-------------|
| `scene_id` | `str` | Scene identifier |
| `detections_path` | `Path` | `detections.geojson` from Stage 3 |
| `processed_dir` | `Path` | Processed scene directory (Stage 2 output) — used to reconstruct the scene data from patches |
| `output_dir` | `Path` | Root output directory (default: `data/detections`) |
| `config` | `dict` | Optional config |

---

## Output

```
data/detections/<SCENE_ID>/
└── detections_classified.geojson   # Original detections + polymer columns
```

**Additional columns added to GeoJSON:**

| Column | Type | Description |
|--------|------|-------------|
| `polymer_type` | `str` | Classified polymer label |
| `pi_value` | `float` | Plastic Index value |
| `sr_value` | `float` | SWIR Ratio value |
| `nsi_value` | `float` | NIR-SWIR Index value |
| `fdi_value` | `float` | Floating Debris Index value |
| `is_false_positive` | `bool` | True if classified as organic matter |

**Returns:** `(Path, Dict[str, int])` — path to classified GeoJSON and polymer count dict.

---

## Band Indices Used

| Variable | Band | Position | Wavelength |
|----------|------|----------|-----------|
| `b04` | B04 (Red) | idx 3 | 665 nm |
| `b06` | B06 (Red Edge) | idx 5 | 740 nm |
| `b08` | B08 (NIR broad) | idx 7 | 832 nm |
| `b8a` | B8A (NIR narrow) | idx 8 | 865 nm |
| `b11` | B11 (SWIR 1) | idx 9 | 1610 nm |

---

## Spectral Indices

### 1. Plastic Index (PI)

Measures NIR-to-Red contrast. Plastic materials have high NIR reflectance relative to red.

```
PI = (B08 - B04) / (B08 + B04 + ε)
```

Where `ε = 1e-8` (numerical stability).

- **High PI (> 0.1):** Strong NIR reflectance → likely plastic
- **Low PI:** Absorbs NIR → likely water or degraded material

---

### 2. SWIR Ratio (SR)

Ratio of SWIR to NIR. Different polymer types have distinct SWIR absorption bands.

```
SR = B11 / (B08 + ε)
```

- **Low SR (< 0.3):** Low SWIR absorption → PE/PP-type polymers
- **High SR (> 0.5):** High SWIR absorption → PET/Nylon-type polymers

---

### 3. NIR-SWIR Index (NSI)

Normalized difference between narrow NIR and SWIR. Organic matter has high NIR reflectance but low SWIR.

```
NSI = (B8A - B11) / (B8A + B11 + ε)
```

- **NSI > 0.2:** Consistent with organic material (seaweed, foam) → false positive flag

---

### 4. Floating Debris Index (FDI)

Detects floating material above the Rayleigh scattering envelope. Derived from spectral interpolation.

```
interpolation = (WL_B08 - WL_RED) / (WL_B11 - WL_RED + ε)
             = (832 - 665) / (1610 - 665)
             = 167 / 945
             ≈ 0.1767

FDI = B08 - (B06 + (B11 - B06) × interpolation)
```

Where wavelengths are:
- `WL_B06 = 740 nm`
- `WL_B08 = 832 nm`
- `WL_B11 = 1610 nm`
- `WL_RED = 665 nm (B04)`

- **FDI > 0.05:** Floating debris signature confirmed by spectral deviation

---

## Polymer Classification Decision Tree

```
IF NSI > 0.2
    → "Organic Matter"         [is_false_positive = True]

ELIF PI > 0.1 AND SR < 0.3
    → "PE/PP (Polyethylene/Polypropylene)"

ELIF PI > 0.1 AND SR > 0.5
    → "PET/Nylon"

ELIF PI > 0.05 AND NSI < 0
    → "Mixed/Degraded Polymer"

ELIF FDI > 0.05
    → "Unidentified Plastic"

ELSE
    → "Unclassified Debris"
```

### Polymer Type Reference

| Type | Common Sources | Key Spectral Signature |
|------|---------------|----------------------|
| PE/PP | Nurdles, packaging, bottles | High PI, low SWIR absorption |
| PET/Nylon | Fishing nets, synthetic ropes | High PI, high SWIR absorption |
| Mixed/Degraded | UV-weathered mixed debris | Moderate PI, negative NSI |
| Unidentified Plastic | Unresolvable polymer | Positive FDI only |
| Organic Matter | Seaweed, foam (false positive) | High NSI |
| Unclassified Debris | Ambiguous spectral signal | All indices low |

---

## Spectral Extraction

To extract a debris cluster's mean spectrum, the normalized patch data is reconstructed from the patch cache:

```python
# Reconstruct full scene from patches (averaging overlaps)
scene_data = np.zeros((11, H, W), dtype=float32)
count       = np.zeros((H, W), dtype=float32)

for patch_id, info in patch_index.items():
    patch = load_patch(patches_dir / info["patch_file"])
    rs, cs = info["row_start"], info["col_start"]
    ah, aw = info["actual_h"], info["actual_w"]
    scene_data[:, rs:rs+ah, cs:cs+aw] += patch[:, :ah, :aw]
    count[rs:rs+ah, cs:cs+aw] += 1.0

scene_data /= max(count, 1)
```

Then for each cluster geometry, `rasterio.features.geometry_mask` is used to create a pixel mask, and the mean spectrum across masked pixels is computed:

```python
mask = geometry_mask([mapping(cluster_geom)], out_shape=(H, W),
                     transform=transform, invert=True)
spectrum = scene_data[:, mask].mean(axis=1)  # shape (11,)
```

---

## Processing Steps

```
1. Load detections.geojson from Stage 3
       ↓
2. Reconstruct full-scene data from Stage 2 patch cache
       ↓
3. For each debris cluster:
     a. Create pixel mask from cluster geometry
     b. Extract mean 11-band spectrum
     c. Compute PI, SR, NSI, FDI
     d. Apply decision tree → polymer_type, is_false_positive
       ↓
4. Append polymer columns to GeoDataFrame
       ↓
5. Save detections_classified.geojson
       ↓
6. Log polymer count summary
```

---

## Fallback Behaviour

If spectral data cannot be extracted (cluster outside scene bounds, patch files missing), the classification falls back to:
```json
{
  "polymer_type": "Unknown (insufficient spectral data)",
  "pi_value": 0.0,
  "sr_value": 0.0,
  "nsi_value": 0.0,
  "fdi_value": 0.0,
  "is_false_positive": false
}
```

---

## Caching

If `detections_classified.geojson` already exists in the output directory, the stage is skipped.

---

## CLI Usage

```bash
python -m pipeline.04_polymer \
    --scene_id     S2A_MSIL2A_20240115... \
    --detections   data/detections/S2A.../detections.geojson \
    --processed_dir data/processed/S2A... \
    --output_dir   data/detections \
    --config       config/config.yaml
```

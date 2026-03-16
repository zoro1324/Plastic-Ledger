# Plastic-Ledger Pipeline Documentation

> End-to-end marine debris detection, classification, and source attribution
> using Sentinel-2 satellite imagery and hydrodynamic modelling.

---

## Table of Contents

- [What the Pipeline Does](#what-the-pipeline-does)
- [Architecture Overview](#architecture-overview)
- [Stage Summary](#stage-summary)
- [Data Flow](#data-flow)
- [Directory Structure](#directory-structure)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [Stage Documentation](#stage-documentation)
- [External APIs at a Glance](#external-apis-at-a-glance)
- [Pipeline Output Formats](#pipeline-output-formats)

---

## What the Pipeline Does

Plastic-Ledger automatically:

1. Downloads satellite imagery of any ocean area you specify
2. Detects marine debris floating on the surface using a deep learning model
3. Identifies what **type of plastic** it likely is from spectral signatures
4. Traces the debris **backward in time** through ocean currents and wind to estimate where it came from
5. Scores candidate source locations against fishing activity, industrial sites, shipping lanes, and rivers
6. Generates a complete **PDF report, GeoJSON, and CSV** with all findings

---

## Architecture Overview

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                        PLASTIC-LEDGER PIPELINE                                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   INPUT                                                                          ║
║  ┌──────────────────────────────────────────────────────────────────────────┐   ║
║  │  bbox (lon_min, lat_min, lon_max, lat_max)  +  target_date               │   ║
║  └──────────────────────────────────────────────────────────────────────────┘   ║
║                                         │                                        ║
║                                         ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────────────┐    ║
║  │  STAGE 1 — Satellite Data Ingestion              pipeline/01_ingest.py  │    ║
║  │                                                                          │    ║
║  │  ● Copernicus STAC API → find best Sentinel-2 L2A scene                 │    ║
║  │  ● OpenID Connect auth → download 8 band GeoTIFFs                       │    ║
║  │  ● Output: data/raw/<SCENE_ID>/{B02,B04,B08...}.tif + metadata.json     │    ║
║  └─────────────────────────────────────────────────────────────────────────┘    ║
║                                         │                                        ║
║                                         ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────────────┐    ║
║  │  STAGE 2 — Preprocessing                       pipeline/02_preprocess.py│    ║
║  │                                                                          │    ║
║  │  ● 8 bands → 11-band model order (zero-pad B01, B06, B07)               │    ║
║  │  ● Clip [0.0001, 0.5] → Z-score normalise (MARIDA stats)                │    ║
║  │  ● Tile into 256×256 patches, 32-pixel overlap                          │    ║
║  │  ● Output: data/processed/<SCENE>/patches/patch_NNNN.npz                │    ║
║  └─────────────────────────────────────────────────────────────────────────┘    ║
║                                         │                                        ║
║                                         ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────────────┐    ║
║  │  STAGE 3 — Marine Debris Detection (Deep Learning)  pipeline/03_detect.py│   ║
║  │                                                                          │    ║
║  │  ● U-Net (ResNet-34) trained on MARIDA → 15-class segmentation          │    ║
║  │  ● Test-Time Augmentation (6 flips/rotations) → averaged probabilities  │    ║
║  │  ● Patch stitching with overlap averaging                                │    ║
║  │  ● Threshold + argmax → debris mask → connected components              │    ║
║  │  ● Output: detections.geojson + debris_mask.tif + debris_prob.tif       │    ║
║  └─────────────────────────────────────────────────────────────────────────┘    ║
║                                         │                                        ║
║                                         ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────────────┐    ║
║  │  STAGE 4 — Polymer Type Classification        pipeline/04_polymer.py    │    ║
║  │                                                                          │    ║
║  │  ● Extract mean 11-band spectrum per cluster                             │    ║
║  │  ● Compute PI, SR, NSI, FDI spectral indices                            │    ║
║  │  ● Rule-based decision tree → PE/PP, PET/Nylon, Mixed, Organic, etc.    │    ║
║  │  ● Flag organic-matter false positives                                   │    ║
║  │  ● Output: detections_classified.geojson                                 │    ║
║  └─────────────────────────────────────────────────────────────────────────┘    ║
║                                         │                                        ║
║                             ┌───────────┴──────────────┐                        ║
║                             ▼                          ▼                         ║
║  ┌──────────────────────────────────────────────────────────────────────┐        ║
║  │  STAGE 5 — Hydrodynamic Back-Tracking       pipeline/05_backtrack.py │        ║
║  │                                                                       │        ║
║  │  ● Download CMEMS ocean currents (uo, vo) + ERA5 wind data           │        ║
║  │  ● Release 50 particles per cluster at detection location            │        ║
║  │  ● RK4 integration backward 30 days at 1-hour steps                  │        ║
║  │  ● DBSCAN-cluster trajectory endpoints → source regions              │        ║
║  │  ● Output: backtrack_summary.json + trajectory GeoJSONs              │        ║
║  └──────────────────────────────────────────────────────────────────────┘        ║
║                                         │                                         ║
║                                         ▼                                         ║
║  ┌──────────────────────────────────────────────────────────────────────┐         ║
║  │  STAGE 6 — Source Attribution               pipeline/06_attribute.py │         ║
║  │                                                                       │         ║
║  │  ● Score each source region on 4 dimensions:                         │         ║
║  │    - Fishing: GFW API vessel activity (weight 0.40)                  │         ║
║  │    - Industrial: OSM waste sites (weight 0.30)                       │         ║
║  │    - Shipping: Shipping lane overlap (weight 0.20)                   │         ║
║  │    - River: Distance to river mouths (weight 0.10)                   │         ║
║  │  ● Weighted composite score → source type + confidence               │         ║
║  │  ● Output: attribution_report.json                                   │         ║
║  └──────────────────────────────────────────────────────────────────────┘         ║
║                                         │                                          ║
║                                         ▼                                          ║
║  ┌──────────────────────────────────────────────────────────────────────┐          ║
║  │  STAGE 7 — Report Generation               pipeline/07_report.py    │          ║
║  │                                                                       │          ║
║  │  ● PDF: executive summary, maps, polymer chart, cluster table         │          ║
║  │  ● GeoJSON: merged detections + attribution                          │          ║
║  │  ● CSV: flat cluster table                                           │          ║
║  │  ● Terminal: rich-formatted summary table                            │          ║
║  └──────────────────────────────────────────────────────────────────────┘          ║
║                                                                                     ║
║   OUTPUT                                                                            ║
║  ┌──────────────────────────────────────────────────────────────────────────────┐  ║
║  │  data/reports/<SCENE>/  →  final_report.pdf  +  final_report.geojson  +      │  ║
║  │                             debris_summary.csv  +  run_summary.json          │  ║
║  └──────────────────────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## Stage Summary

| # | Stage | What it does | Key Tech | Output |
|---|-------|-------------|----------|--------|
| 1 | **Ingest** | Download satellite imagery | Copernicus STAC API, pystac-client | Raw GeoTIFF bands |
| 2 | **Preprocess** | Normalise and tile bands into patches | rasterio, numpy | 256×256 `.npz` patches |
| 3 | **Detect** | Segment debris from non-debris | U-Net + ResNet-34, PyTorch, TTA | `detections.geojson` |
| 4 | **Polymer** | Classify debris type from spectrum | Spectral indices (PI, SR, NSI, FDI) | `detections_classified.geojson` |
| 5 | **Backtrack** | Trace debris origin via particle modelling | CMEMS, ERA5, RK4 integration, DBSCAN | Trajectory GeoJSONs + source regions |
| 6 | **Attribute** | Score sources against human activities | GFW API, OSM, shipping lanes | `attribution_report.json` |
| 7 | **Report** | Assemble final reports | fpdf2, matplotlib | PDF, GeoJSON, CSV |

---

## Data Flow

```
SATELLITE IMAGE
(Sentinel-2 L2A, 10–20 m, 8 bands)
        │
        │ Stage 1: Download
        ▼
RAW BANDS (GeoTIFF per band)
        │
        │ Stage 2: Band assembly, Z-score normalisation, tiling
        ▼
PATCHES (11 × 256 × 256, float16, ~1000s per scene)
        │
        │ Stage 3: U-Net inference + TTA + stitch + threshold
        ▼
DEBRIS PROBABILITIES (float32, same spatial extent)
+ DEBRIS MASK (binary, uint8)
+ DETECTION CLUSTERS (GeoJSON polygons)
        │
        │ Stage 4: Spectral extraction → PI/SR/NSI/FDI → decision tree
        ▼
CLASSIFIED DETECTIONS (GeoJSON + polymer_type column)
        │
        ├─────────────────────────────────────────────┐
        │                                             │
        │ Stage 5: Particle release +                 │
        │ RK4 back-tracking + DBSCAN                  │
        ▼                                             │
SOURCE REGIONS (JSON list)                           │
        │                                             │
        │ Stage 6: GFW + OSM + shipping + rivers      │
        ▼                                             │
ATTRIBUTION REPORT (JSON, ranked by score)           │
        │                                             │
        └─────────────────────┬───────────────────────┘
                              │ Stage 7: Merge + render
                              ▼
                  FINAL REPORT (PDF + GeoJSON + CSV)
```

---

## Directory Structure

```
data/
├── raw/
│   └── <SCENE_ID>/
│       ├── B02.tif  B03.tif  B04.tif  B05.tif
│       ├── B08.tif  B8A.tif  B11.tif  B12.tif
│       └── metadata.json
│
├── processed/
│   └── <SCENE_ID>/
│       ├── patches/
│       │   ├── patch_0000.npz
│       │   └── ...
│       ├── patch_index.json
│       ├── nodata_mask.npy
│       └── scene_meta.json
│
├── detections/
│   └── <SCENE_ID>/
│       ├── detections.geojson              ← Stage 3
│       ├── detections_classified.geojson   ← Stage 4
│       ├── debris_mask.tif
│       ├── debris_prob.tif
│       └── class_mask.tif
│
├── attribution/
│   └── <SCENE_ID>/
│       ├── backtrack_summary.json          ← Stage 5
│       ├── backtrack_<id>.geojson
│       ├── attribution_report.json         ← Stage 6
│       └── forcing_data/
│           ├── ocean_currents.nc
│           └── wind_data.nc
│
└── reports/
    └── <SCENE_ID>/
        ├── final_report.pdf                ← Stage 7
        ├── final_report.geojson
        ├── debris_summary.csv
        ├── detection_map.png
        └── polymer_distribution.png
```

---

## Quick Start

### Run the full pipeline

```bash
python pipeline/run_pipeline.py \
    --bbox         "80.0,8.0,82.0,10.0" \
    --date         "2024-01-31" \
    --output_dir   "data/runs/run_001" \
    --model_path   "best_model/best_model.pth" \
    --cloud_cover  20 \
    --backtrack_days 30
```

### Run individual stages

```bash
# Stage 1 only
python -m pipeline.01_ingest --bbox "80,8,82,10" --start_date 2024-01-01 --end_date 2024-01-31

# Stage 2 only
python -m pipeline.02_preprocess --scene_dir data/raw/S2A_...

# Stage 3 only
python -m pipeline.03_detect --scene_id S2A_... --patches_dir data/processed/S2A_.../patches --model_path best_model/best_model.pth

# Stage 4 only
python -m pipeline.04_polymer --scene_id S2A_... --detections data/detections/S2A_.../detections.geojson --processed_dir data/processed/S2A_...

# Stage 5 only
python -m pipeline.05_backtrack --scene_id S2A_... --detections data/detections/S2A_.../detections_classified.geojson

# Stage 6 only
python -m pipeline.06_attribute --scene_id S2A_... --sources data/attribution/S2A_.../backtrack_summary.json --detections data/detections/S2A_.../detections_classified.geojson

# Stage 7 only
python -m pipeline.07_report --scene_id S2A_... --detections data/detections/S2A_.../detections_classified.geojson --attribution data/attribution/S2A_.../attribution_report.json
```

### Skip specific stages

```bash
python pipeline/run_pipeline.py \
    --bbox "80,8,82,10" --date 2024-01-31 \
    --skip_stages "1,5"    # Skip ingest and backtracking
```

---

## Configuration Reference

All settings live in `config/config.yaml`:

```yaml
model:
  checkpoint: models/runs/marida_v1/best_model.pth
  num_bands: 11
  num_classes: 15
  encoder: resnet34
  debris_threshold: 0.1          # Min probability to flag pixel as debris
  tta: true                       # Enable Test-Time Augmentation

preprocessing:
  patch_size: 256                 # Patch dimensions (pixels)
  overlap: 32                     # Patch overlap (pixels)
  patch_storage: npz              # npy = raw, npz = compressed
  patch_dtype: float16            # float32 or float16

detection:
  min_cluster_area_m2: 100        # Filter clusters smaller than this
  debris_class_index: 0

backtracking:
  days: 30                        # How far to integrate backward
  n_particles: 50                 # Particles per cluster
  time_step_hours: 1              # RK4 integration step
  ocean_wind_ratio: [0.97, 0.03]  # Ocean vs wind drift weights
  dbscan_eps_degrees: 0.5         # DBSCAN epsilon (degrees)
  dbscan_min_samples: 5           # DBSCAN min cluster size

attribution:
  weights:
    fishing: 0.4                  # GFW fishing activity weight
    industrial: 0.3               # OSM industrial sites weight
    shipping: 0.2                 # Shipping lane overlap weight
    river: 0.1                    # River mouth proximity weight
  search_radius_km: 10

apis:
  copernicus_username: ${COPERNICUS_USERNAME}
  copernicus_password: ${COPERNICUS_PASSWORD}
  gfw_token: ${GFW_TOKEN}
```

Environment variables are loaded from `.env`.

---

## Stage Documentation

| Stage | Document Link |
|-------|--------------|
| 1 — Satellite Data Ingestion | [stage-01-ingest.md](stage-01-ingest.md) |
| 2 — Preprocessing | [stage-02-preprocess.md](stage-02-preprocess.md) |
| 3 — Marine Debris Detection | [stage-03-detect.md](stage-03-detect.md) |
| 4 — Polymer Type Classification | [stage-04-polymer.md](stage-04-polymer.md) |
| 5 — Hydrodynamic Back-Tracking | [stage-05-backtrack.md](stage-05-backtrack.md) |
| 6 — Source Attribution | [stage-06-attribute.md](stage-06-attribute.md) |
| 7 — Report Generation | [stage-07-report.md](stage-07-report.md) |

---

## External APIs at a Glance

| API | Stage | Purpose | Authentication |
|-----|-------|---------|---------------|
| Copernicus STAC | 1 | Search Sentinel-2 scenes | Public (no auth needed for search) |
| Copernicus Identity | 1 | Download imagery | OpenID Connect (username/password) |
| CMEMS (Copernicus Marine) | 5 | Ocean current data | `copernicusmarine` CLI auth |
| ERA5 / CDS API | 5 | Wind data | CDS API key (`.cdsapirc`) |
| Global Fishing Watch | 6 | Fishing vessel activity | Bearer token (`GFW_TOKEN`) |
| OpenStreetMap (via osmnx) | 6 | Industrial/waste sites | Public (no auth) |

---

## Pipeline Output Formats

### `detections.geojson` (GeoJSON FeatureCollection)
- **CRS:** EPSG:4326
- **Geometry:** Polygon (debris cluster outline)
- **Properties:** `cluster_id`, `area_m2`, `mean_confidence`, `centroid_lon`, `centroid_lat`, `detection_date`

### `detections_classified.geojson` (extends detections)
- **Additional properties:** `polymer_type`, `pi_value`, `sr_value`, `nsi_value`, `fdi_value`, `is_false_positive`

### `attribution_report.json`
- **Format:** JSON array, sorted descending by `attribution_score`
- **Key fields:** `source_type`, `attribution_score`, `confidence`, `explanation`, `source_centroid`, `vessel_ids`

### `final_report.geojson`
- **Format:** Merged detections + attribution — ready for QGIS/ArcGIS

### `debris_summary.csv`
- **Format:** One row per cluster, flat structure — ready for spreadsheet analysis

### `run_summary.json`
- **Format:** Pipeline run metadata — stages completed/failed, elapsed time, output paths

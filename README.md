# 🌊 Plastic-Ledger

**Autonomous Micro-Plastic Fingerprinting & Source Attribution from Satellite Imagery**

Plastic-Ledger is an end-to-end Python pipeline that detects marine plastic debris in Sentinel-2
satellite imagery, classifies the polymer type, traces debris back to its source using ocean
current simulations, and generates comprehensive attribution reports. It uses a U-Net deep
learning model trained on the [MARIDA](https://github.com/marine-debris/marine-debris.github.io)
dataset for segmentation, combined with spectral analysis, Lagrangian particle tracking, and
multi-source geospatial attribution.

---

## 🔬 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PLASTIC-LEDGER PIPELINE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ Stage 1  │───▶│ Stage 2  │───▶│ Stage 3  │───▶│ Stage 4  │      │
│  │ Ingest   │    │Preprocess│    │ Detect   │    │ Polymer  │      │
│  │          │    │          │    │          │    │ Classify │      │
│  │ Sentinel │    │ Band     │    │ U-Net +  │    │ Spectral │      │
│  │ 2 STAC   │    │ Reorder  │    │ TTA      │    │ Indices  │      │
│  │ Download │    │ Normalize│    │ Cluster  │    │ Rules    │      │
│  └──────────┘    │ Tile     │    └──────────┘    └─────┬────┘      │
│                  └──────────┘                          │            │
│                                                        ▼            │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ Stage 7  │◀───│ Stage 6  │◀───│ Stage 5  │◀───│          │      │
│  │ Report   │    │Attribute │    │Backtrack │    │ Debris   │      │
│  │          │    │          │    │          │    │ Clusters │      │
│  │ PDF      │    │ Fishing  │    │ CMEMS +  │    │ + Polymer│      │
│  │ GeoJSON  │    │ Industry │    │ ERA5     │    │ Type     │      │
│  │ CSV      │    │ Shipping │    │ RK4      │    └──────────┘      │
│  │ Terminal │    │ Rivers   │    │ DBSCAN   │                      │
│  └──────────┘    └──────────┘    └──────────┘                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-org/Plastic-Ledger.git
cd Plastic-Ledger

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API credentials
cp .env.example .env
# Edit .env with your API keys (see "API Keys" section below)

# 4. Run the pipeline on a test area (Sri Lanka coast)
python pipeline/run_pipeline.py \
    --bbox "80.5,7.5,81.5,8.5" \
    --start_date "2024-01-10" \
    --end_date "2024-01-15" \
    --output_dir "data/runs/test_run" \
    --model_path "models/runs/marida_v1/best_model.pth" \
    --cloud_cover 20 \
    --backtrack_days 7

# 5. Check output
ls data/runs/test_run/reports/
```

---

## 🔑 API Keys Setup

The pipeline requires credentials for external data sources. All keys are stored in `.env`:

| Service | Purpose | Sign Up |
|---------|---------|---------|
| **Copernicus Data Space** | Sentinel-2 satellite imagery download | [dataspace.copernicus.eu](https://dataspace.copernicus.eu) (free) |
| **CMEMS** | Ocean current data (surface velocity) | [marine.copernicus.eu](https://marine.copernicus.eu) (free) |
| **CDS API** | ERA5 wind data for drift calculation | [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu) (free) |
| **Global Fishing Watch** | Fishing vessel positions | [globalfishingwatch.org](https://globalfishingwatch.org/data/) (free academic) |

```bash
# .env file format:
COPERNICUS_USERNAME=your_email@example.com
COPERNICUS_PASSWORD=your_password
GFW_TOKEN=your_global_fishing_watch_token
CDS_API_KEY=your_climate_data_store_key
```

> **Note**: The pipeline runs in graceful degradation mode — if API keys are missing, it will
> use heuristic fallbacks for scoring (Stages 5 and 6) and skip data downloads.

---

## 📋 Example CLI Command

```bash
# Full pipeline — Sri Lanka coast, January 2024
python pipeline/run_pipeline.py \
    --bbox "80.0,6.0,82.0,8.0" \
    --start_date "2024-01-01" \
    --end_date "2024-01-31" \
    --output_dir "data/runs/sri_lanka_jan24" \
    --model_path "models/runs/marida_v1/best_model.pth" \
    --cloud_cover 20 \
    --backtrack_days 30

# Skip stages 1 and 2 (if data is already downloaded & preprocessed)
python pipeline/run_pipeline.py \
    --bbox "80.0,6.0,82.0,8.0" \
    --start_date "2024-01-01" \
    --end_date "2024-01-31" \
    --output_dir "data/runs/sri_lanka_jan24" \
    --skip_stages "1,2"
```

Each stage can also be run independently:

```bash
# Run just Stage 3 (detection) on pre-existing patches
python -m pipeline.03_detect \
    --scene_id S2A_MSIL2A_20240115 \
    --patches_dir data/processed/S2A_MSIL2A_20240115/patches \
    --model_path models/runs/marida_v1/best_model.pth
```

---

## 📁 Output Files

| File | Description |
|------|-------------|
| `final_report.pdf` | Executive summary with detection maps, polymer charts, and attribution tables |
| `final_report.geojson` | All detections with attribution data as GeoJSON (for GIS tools) |
| `debris_summary.csv` | Flat CSV with one row per debris cluster |
| `debris_mask.tif` | Binary GeoTIFF mask of detected debris pixels |
| `debris_prob.tif` | Float32 GeoTIFF probability map of debris class |
| `class_mask.tif` | Full 15-class prediction mask (uint8 GeoTIFF) |
| `detections.geojson` | Raw debris detection polygons with area and confidence |
| `detections_classified.geojson` | Detections with polymer type classification |
| `backtrack_*.geojson` | Particle trajectories per debris cluster |
| `attribution_report.json` | Source attribution scores and explanations |
| `run_summary.json` | Pipeline run metadata and timing |

---

## 🏗️ Project Structure

```
Plastic-Ledger/
├── models/
│   └── runs/marida_v1/
│       └── best_model.pth              ← Trained U-Net (do not modify)
├── pipeline/
│   ├── 01_ingest.py                    ← Sentinel-2 STAC search & download
│   ├── 02_preprocess.py                ← Band reorder, normalize, tile
│   ├── 03_detect.py                    ← U-Net inference + TTA + clustering
│   ├── 04_polymer.py                   ← Spectral index polymer classification
│   ├── 05_backtrack.py                 ← Lagrangian RK4 particle tracking
│   ├── 06_attribute.py                 ← Multi-source attribution scoring
│   ├── 07_report.py                    ← PDF + GeoJSON + CSV + terminal output
│   ├── run_pipeline.py                 ← Master pipeline orchestrator
│   └── utils/
│       ├── logging_utils.py            ← Rich-based logger
│       ├── geo_utils.py                ← GeoTIFF I/O, polygon conversion
│       └── cache_utils.py              ← Stage caching, config loader
├── config/
│   └── config.yaml                     ← All thresholds and parameters
├── tests/
│   ├── test_stage_1.py … test_stage_7.py
├── data/                               ← Generated outputs (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚠️ Known Limitations

1. **Model accuracy**: The MARIDA-trained U-Net has a debris IoU of ~0.03 on the test set.
   This is typical for rare-class marine debris detection — the model is more useful as a
   screening tool than a precision detector.

2. **Band coverage**: Only 8 of 11 Sentinel-2 bands are downloaded (B01, B06, B07 are
   zero-padded). This may affect polymer classification accuracy for those indices that
   depend on B06 (used in FDI).

3. **Backtracking data**: CMEMS and ERA5 require active accounts. Without credentials,
   the pipeline uses synthetic random velocities for backtracking (useful for testing
   the pipeline but not for real analysis).

4. **Source attribution**: The attribution stage uses heuristic scoring when reference
   datasets (shipping lanes, river mouths) are not locally cached. Download reference
   data to `data/reference/` for more accurate results.

5. **10m resolution**: Debris smaller than ~100m² (4 pixels) is filtered out. Individual
   nurdles or small debris items are below the detection limit.

6. **Cloud cover**: Sentinel-2 is optical — cloudy scenes cannot be analyzed. Use the
   `--cloud_cover` flag to filter scenes by maximum cloud percentage.

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

Tests use mocked data and do not require API keys, GPU, or downloaded imagery.

---

## 📄 License

This project is for research and educational purposes. The MARIDA dataset and Sentinel-2
imagery are subject to their respective licenses (Copernicus Data Space EULA).

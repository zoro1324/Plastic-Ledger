# Stage 1: Satellite Data Ingestion

**File:** `pipeline/01_ingest.py`

---

## Overview

Stage 1 is responsible for discovering and downloading Sentinel-2 L2A satellite imagery from the **Copernicus Data Space Ecosystem** via its STAC (SpatioTemporal Asset Catalog) API. It takes a geographic bounding box and date range as input and outputs raw per-band GeoTIFF files on disk.

---

## Input

| Parameter | Type | Description |
|-----------|------|-------------|
| `bbox` | `(lon_min, lat_min, lon_max, lat_max)` | Geographic region of interest in WGS-84 degrees |
| `date_start` | `str` (ISO 8601) | Start of the search window, e.g. `2024-01-01` |
| `date_end` | `str` (ISO 8601) | End of the search window / target date |
| `cloud_cover_max` | `int` (0–100) | Maximum acceptable cloud cover percentage (default: `20`) |
| `output_dir` | `Path` | Root directory for downloaded raw scenes (default: `data/raw`) |
| `config` | `dict` | Optional YAML config with API credentials |

---

## Output

```
data/raw/
└── <SCENE_ID>/
    ├── B02.tif          # Blue band (10 m)
    ├── B03.tif          # Green band (10 m)
    ├── B04.tif          # Red band (10 m)
    ├── B05.tif          # Red Edge 1 (20 m)
    ├── B08.tif          # NIR broad (10 m)
    ├── B8A.tif          # NIR narrow (20 m)
    ├── B11.tif          # SWIR 1 (20 m)
    ├── B12.tif          # SWIR 2 (20 m)
    └── metadata.json    # Scene metadata (datetime, bbox, cloud cover, etc.)
data/raw/ingest_metadata.json   # Run-level summary
```

**Returns:** `(List[Path], List[Dict])` — list of scene directories and scene metadata dicts.

---

## APIs Used

### Copernicus STAC API

| Property | Value |
|----------|-------|
| Endpoint | `https://catalogue.dataspace.copernicus.eu/stac` |
| Library | `pystac-client` |
| Collection | `sentinel-2-l2a` |
| Auth | Bearer token (OpenID Connect) |

**Search query parameters:**
```python
catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime=f"{date_start}/{date_end}",
    query={"eo:cloud_cover": {"lt": cloud_cover_max}},
    sortby=[{"field": "datetime", "direction": "desc"}],
    max_items=1,   # Only the single most-recent qualifying scene
)
```

### Copernicus Identity (Authentication)

| Property | Value |
|----------|-------|
| Token URL | `https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token` |
| Grant Type | `password` |
| Client ID | `cdse-public` |

Authentication produces a Bearer token attached to every subsequent download request.

---

## Key Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `STAC_ENDPOINT` | `https://catalogue.dataspace.copernicus.eu/stac` | STAC catalog base URL |
| `COLLECTION` | `sentinel-2-l2a` | Sentinel-2 Level-2A surface reflectance |
| `REQUIRED_BANDS` | `B02, B03, B04, B05, B08, B8A, B11, B12` | 8 spectral bands that feed into the model |

---

## Processing Steps

```
1. Open STAC catalog connection
       ↓
2. Run date/bbox/cloud-cover query
       ↓
3. Pick the single newest qualifying scene
       ↓
4. Authenticate via OpenID Connect (if credentials provided)
       ↓
5. For each required band:
       - Resolve STAC asset URL (handles keys like "B02_10m", "B02", etc.)
       - Stream-download to data/raw/<SCENE_ID>/<BAND>.tif
       ↓
6. Write metadata.json alongside the bands
       ↓
7. Write data/raw/ingest_metadata.json for pipeline run logging
```

---

## Retry Logic

All network calls are wrapped with `@retry_request` (from `pipeline.utils.geo_utils`), which performs exponential back-off retries on transient HTTP errors (5xx, timeouts).

---

## Caching

If a band file already exists at the destination path, the download is skipped — no re-downloading on repeated runs.

---

## Sentinel-2 Band Reference

| Band | Wavelength (nm) | Resolution | Usage in pipeline |
|------|----------------|------------|-------------------|
| B02 | 490 (Blue) | 10 m | Polymer spectral index |
| B03 | 560 (Green) | 10 m | Visual / spectral |
| B04 | 665 (Red) | 10 m | PI formula |
| B05 | 705 (Red Edge 1) | 20 m | Spectral analysis |
| B08 | 842 (NIR broad) | 10 m | PI, FDI, NSI formulas |
| B8A | 865 (NIR narrow) | 20 m | NSI formula |
| B11 | 1610 (SWIR 1) | 20 m | SR, NSI, FDI formulas |
| B12 | 2190 (SWIR 2) | 20 m | Supplemental spectral |

> **Note:** B01, B06, B07 are not downloaded — they are zero-padded in Stage 2 when assembling the 11-band model input.

---

## Error Conditions

| Error | Cause | Behaviour |
|-------|-------|-----------|
| `RuntimeError: No scenes found` | No image satisfies bbox/date/cloud filters | Pipeline aborts |
| `RuntimeError: All downloads failed` | Every band download raised an exception | Pipeline aborts |
| `KeyError: band not in assets` | STAC item missing expected band key | Band skipped with warning |
| Auth failure | Wrong credentials or network issue | Falls back to unauthenticated download |

---

## CLI Usage

```bash
python -m pipeline.01_ingest \
    --bbox "80.0,8.0,82.0,10.0" \
    --start_date 2024-01-01 \
    --end_date   2024-01-31 \
    --cloud_cover 20 \
    --output_dir data/raw \
    --config config/config.yaml
```

---

## Config Keys Used

```yaml
apis:
  copernicus_username: ${COPERNICUS_USERNAME}  # Copernicus Data Space email
  copernicus_password: ${COPERNICUS_PASSWORD}  # Copernicus Data Space password
```

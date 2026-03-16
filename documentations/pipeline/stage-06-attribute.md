# Stage 6: Source Attribution

**File:** `pipeline/06_attribute.py`

---

## Overview

Stage 6 takes the candidate source regions from Stage 5 (back-tracked particle endpoints) and **scores them against four pollution source categories**: fishing activity, industrial/waste sites, major shipping lanes, and river mouths. A weighted composite score determines the most probable pollution source type, and a human-readable explanation is generated.

---

## Input

| Parameter | Type | Description |
|-----------|------|-------------|
| `scene_id` | `str` | Scene identifier |
| `sources` | `List[Dict]` | Source region list from Stage 5 `backtrack_summary.json` |
| `detections_path` | `Path` | `detections_classified.geojson` from Stage 4 |
| `output_dir` | `Path` | Root output directory (default: `data/attribution`) |
| `config` | `dict` | Weights, radius, API keys |
| `detection_date` | `str` (ISO 8601) | Detection date (for time-bounded API queries) |

---

## Output

```
data/attribution/<SCENE_ID>/
└── attribution_report.json   # Ranked list of attributed source regions
```

**Returns:** `Path` to `attribution_report.json`

---

## Attribution Report Schema

Each entry in `attribution_report.json`:
```json
{
  "debris_cluster_id":  0,
  "source_rank":        1,
  "source_type":        "fishing",
  "location_name":      "Sri Lankan Coast",
  "country":            "Sri Lanka",
  "attribution_score":  0.78,
  "confidence":         "high",
  "explanation":        "High probability (78%): Fishing activity near ...",
  "source_centroid":    [80.5, 8.2],
  "source_bbox":        [80.1, 7.9, 80.9, 8.5],
  "source_probability": 0.72,
  "days_to_source":     30,
  "fishing_score":      0.9,
  "industrial_score":   0.3,
  "shipping_score":     0.6,
  "river_score":        0.1,
  "vessel_ids":         ["vessel_abc123"]
}
```

---

## Scoring Dimensions

### 1. Fishing Vessel Activity Score

**API:** Global Fishing Watch (GFW) REST API v3

```
Endpoint: https://gateway.api.globalfishingwatch.org/v3/4wings/report
Auth: Bearer <gfw_token>
Params:
  spatial-resolution: low
  temporal-resolution: monthly
  group-by: flag
  datasets[0]: public-global-fishing-effort:latest
  date-range: <date_start>,<date_end>
  geometry: GeoJSON Polygon (expanded bbox)
```

**Score formula:**
```
vessel_count = len(response["entries"])
score        = min(1.0, vessel_count / 20.0)
```
A source region with 20 or more fishing vessels within 30 days scores 1.0.

**Heuristic fallback (no GFW token):**
```
score = max(0, 1.0 - abs(centroid_lat) / 60.0) × 0.5
```
Fishing is assumed more common at tropical/subtropical latitudes.

---

### 2. Industrial / Waste Site Score

**API:** OpenStreetMap via `osmnx`

Tags queried within the `search_radius_km` expanded bbox:
```python
tags = {
    "amenity": ["waste_disposal", "waste_transfer_station", "recycling"],
    "landuse": ["industrial"],
    "man_made": ["wastewater_plant"],
}
gdf = ox.features_from_bbox(bbox=expanded, tags=tags)
```

**Score formula:**
```
site_count = len(gdf)
score      = min(1.0, site_count / 10.0)
```
10 or more sites within range scores 1.0.

**Heuristic fallback (osmnx unavailable):** `score = 0.2`

---

### 3. Shipping Lane Score

**Data source:** Local GeoJSON file `data/reference/shipping_lanes.geojson` (if available)

```
overlap_area = intersection(source_bbox, shipping_lanes).area
score        = min(1.0, overlap_area / source_area)
```

**Heuristic fallback (no shipping lane file):**
```python
is_near_shipping = (
    (5 < lat < 25  and 60  < lon < 120) or  # Indian Ocean
    (0 < lat < 40  and -10 < lon < 40 ) or  # Mediterranean
    (20 < lat < 50 and 100 < lon < 180)      # Pacific
)
score = 0.6 if is_near_shipping else 0.2
```

---

### 4. River Discharge Score

**Data source:** Local GeoJSON file `data/reference/river_mouths.geojson` (if available)

```
min_distance_km = min(distance_degrees_to_each_river × 111 km/°)
score           = max(0, 1.0 - min_distance_km / max_distance_km)
```
Where `max_distance_km = 200`.

**Heuristic fallback (no river mouths file):** Pre-coded list of major river mouths:

| River | Longitude | Latitude |
|-------|-----------|---------|
| Ganges | 88.8 | 21.7 |
| Indus | 67.5 | 23.9 |
| Mekong | 106.7 | 9.8 |
| Yangtze | 121.9 | 31.4 |
| Nile | 31.5 | 31.5 |
| Niger | 6.0 | 4.3 |
| Amazon | -50.0 | -0.5 |
| Pearl | 113.5 | 22.2 |
| Mahaweli | 81.3 | 8.6 |
| Kelani | 79.8 | 6.9 |

Distance formula:
```
dist_km = sqrt(
    (Δlon × 111 × cos(lat))² + (Δlat × 111)²
)
```

---

## Composite Attribution Score

The four dimension scores are combined into a single **weighted composite**:

```
attribution_score = Σ (weight[k] × score[k])
```

**Default weights (from config):**

| Source Type | Weight |
|-------------|--------|
| `fishing` | 0.40 |
| `industrial` | 0.30 |
| `shipping` | 0.20 |
| `river` | 0.10 |
| **Total** | **1.00** |

**Source type determination:**
```python
source_type = max(weights.keys(), key=lambda k: scores[k]["score"])
```

**Confidence mapping:**
```
attribution_score > 0.6  → "high"
attribution_score > 0.3  → "moderate"
else                     → "low"
```

---

## Explanation Generation

A human-readable explanation is generated based on the dominant source type:

```
"High probability (78%): Fishing activity near (81.23, 8.45).
 12 vessel(s) detected in this area ~30 days before detection."

"Moderate probability (45%): River discharge from Ganges (180 km from
 source region). Estimated 30 days drift time."
```

---

## Processing Steps

```
1. Load source regions from Stage 5 backtrack_summary.json
       ↓
2. Determine 30-day date window ending at detection_date
       ↓
3. For each source region:
     a. Score fishing: GFW API (or heuristic fallback)
     b. Score industrial: OSM query (or heuristic fallback)
     c. Score shipping: GeoJSON overlay (or lat/lon heuristic)
     d. Score river: River mouths distance (or pre-coded lookup)
     e. Compute weighted composite score
     f. Determine source_type = highest individual score
     g. Assign confidence label
     h. Generate explanation text
       ↓
4. Sort entries descending by attribution_score
       ↓
5. Save attribution_report.json
```

---

## Special Cases

- **Polar latitudes (abs(lat) ≥ 80°):** GFW and OSM queries are skipped; heuristic scores are used instead
- **GFW 422 error (Unprocessable Entity):** GFW is disabled for all subsequent sources in the same run (API geometry rejection)
- **Empty source list:** An empty `attribution_report.json` is written

---

## Caching

If `attribution_report.json` already exists, the stage is skipped.

---

## CLI Usage

```bash
python -m pipeline.06_attribute \
    --scene_id        S2A_MSIL2A_20240115... \
    --sources         data/attribution/S2A.../backtrack_summary.json \
    --detections      data/detections/S2A.../detections_classified.geojson \
    --output_dir      data/attribution \
    --detection_date  2024-01-15 \
    --config          config/config.yaml
```

---

## Config Keys Used

```yaml
attribution:
  weights:
    fishing:    0.4
    industrial: 0.3
    shipping:   0.2
    river:      0.1
  search_radius_km: 10   # Expansion radius for fishing/industrial queries

apis:
  gfw_token: ${GFW_TOKEN}  # Global Fishing Watch API Bearer token
```

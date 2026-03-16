# Stage 7: Report Generation

**File:** `pipeline/07_report.py`

---

## Overview

Stage 7 is the final output stage. It assembles all results from the previous stages and generates four report artefacts: a **multi-page PDF report**, a **combined GeoJSON summary**, a **flat CSV export**, and a **rich terminal summary table** printed to stdout.

---

## Input

| Parameter | Type | Description |
|-----------|------|-------------|
| `scene_id` | `str` | Scene identifier |
| `detections_path` | `Path` | `detections_classified.geojson` from Stage 4 |
| `attribution_path` | `Path` | `attribution_report.json` from Stage 6 |
| `output_dir` | `Path` | Root output directory (default: `data/reports`) |
| `config` | `dict` | Optional config |

---

## Output

```
data/reports/<SCENE_ID>/
├── final_report.pdf          # Multi-page PDF report
├── final_report.geojson      # Detection polygons with attribution merged
├── debris_summary.csv        # Flat CSV of all clusters
├── detection_map.png         # Debris cluster map image
└── polymer_distribution.png  # Pie chart of polymer types
```

**Returns:** `Dict[str, Path]` mapping `{pdf, geojson, csv}` → file paths.

---

## PDF Report Structure

Generated with `fpdf2`.

### Page 1 — Executive Summary

| Field | Source |
|-------|--------|
| Detection Date | `detections_gdf["detection_date"]` |
| Total Debris Clusters | `len(detections_gdf)` |
| Total Debris Area (km²) | `sum(area_m2) / 1,000,000` |
| Dominant Polymer Type | Most frequent `polymer_type` value |
| Top Source Attribution | First entry from `attribution_report.json` |

Followed by the **detection map image** (`detection_map.png`).

### Page 2 — Polymer Distribution

A pie chart showing the count breakdown of all polymer types across detected clusters.

### Page 3 — Source Attribution (Top 3)

For each of the top 3 attribution entries:
- Source type and confidence score
- Human-readable explanation from Stage 6

### Page 4 — Back-Track Trajectories

Trajectory map image (if generated externally).

### Page 5 — Cluster Detail Table

A tabular listing of all debris clusters:

| Column | Width | Description |
|--------|-------|-------------|
| ID | 15 | `cluster_id` |
| Area (sq m) | 25 | `area_m2` |
| Confidence | 20 | `mean_confidence` |
| Polymer | 40 | `polymer_type` |
| Lat | 25 | `centroid_lat` |
| Lon | 30 | `centroid_lon` |
| Source | 30 | Attribution `source_type` |

---

## Detection Map

A Matplotlib figure with:
- **Background:** Dark navy (`#1a1a2e`)
- **Debris clusters:** Red polygons (`#E63946`, 70% opacity)
- All geometry is plotted in raw data coordinates to avoid projection artefacts

```python
plot_gdf.plot(ax=ax, color="#E63946", alpha=0.7, edgecolor="white", linewidth=0.5)
```

Saved as 150 DPI PNG.

---

## Polymer Pie Chart

A Matplotlib pie chart on a dark background:
- Per-type count is labelled with percentage
- Fixed colour palette: `["#E63946", "#2A9D8F", "#F4A261", "#3A86FF", "#8338EC", ...]`

```python
ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", ...)
```

---

## Combined GeoJSON

The final `final_report.geojson` merges detection geometry with attribution data:

**Additional columns merged from `attribution_report.json`:**
- `source_type`
- `attribution_score`
- `explanation`
- `country`

Merge key: `cluster_id` (detection) matched to `debris_cluster_id` (attribution).

---

## CSV Schema

`debris_summary.csv` — one row per debris cluster:

| Column | Source |
|--------|--------|
| `cluster_id` | Detection `cluster_id` |
| `lat` | Detection `centroid_lat` |
| `lon` | Detection `centroid_lon` |
| `area_sq_m` | Detection `area_m2` |
| `polymer_type` | Stage 4 classification |
| `confidence` | Stage 3 `mean_confidence` |
| `top_source_type` | Stage 6 `source_type` |
| `top_source_location` | Stage 6 `location_name` |
| `top_source_country` | Stage 6 `country` |
| `attribution_score` | Stage 6 `attribution_score` |
| `detection_date` | Stage 1 `datetime` |
| `scene_id` | Scene identifier |

---

## Terminal Summary (Rich)

If the `rich` library is installed, a formatted terminal output is produced:

```
╭──────────────────────────────────────────╮
│   Plastic-Ledger Report — Scene: S2A_... │
╰──────────────────────────────────────────╯

  Debris Clusters: 5
  Total Area: 12500 m² (0.0125 km²)

┌─────────────────────────────────────────────────────────────┐
│                 Detected Debris Clusters                    │
├──────┬────────────┬─────────────┬────────────────┬─────────┤
│  ID  │ Area (sq m)│  Confidence │ Polymer Type   │Location │
└──────┴────────────┴─────────────┴────────────────┴─────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Source Attribution                       │
├──────┬────────────┬──────┬──────────────┬──────────────────┤
│ Rank │ Source Type│Score │   Location   │  Explanation     │
└──────┴────────────┴──────┴──────────────┴──────────────────┘
```

Scores are colour-coded: green (>60%), yellow (>30%), red (<30%).

If `rich` is not installed, a plain-text fallback is printed instead.

---

## Processing Steps

```
1. Load detections_classified.geojson
       ↓
2. Load attribution_report.json
       ↓
3. Generate detection_map.png (scatter of cluster polygons)
       ↓
4. Generate polymer_distribution.png (pie chart)
       ↓
5. Generate PDF:
     - Page 1: Summary + detection map
     - Page 2: Polymer chart
     - Page 3: Top-3 attribution explanations
     - Page 4: Cluster table
       ↓
6. Generate final_report.geojson (detections + attribution merge)
       ↓
7. Generate debris_summary.csv (flat table)
       ↓
8. Print rich terminal summary
```

---

## Caching

If all three of `final_report.pdf`, `final_report.geojson`, and `debris_summary.csv` already exist, the stage is skipped.

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `fpdf2` | PDF generation |
| `matplotlib` | Charts and maps |
| `geopandas` | GeoJSON reading and writing |
| `pandas` | CSV export |
| `rich` | Terminal table formatting (optional) |

---

## CLI Usage

```bash
python -m pipeline.07_report \
    --scene_id   S2A_MSIL2A_20240115... \
    --detections data/detections/S2A.../detections_classified.geojson \
    --attribution data/attribution/S2A.../attribution_report.json \
    --output_dir  data/reports \
    --config      config/config.yaml
```

# Stage 5: Hydrodynamic Back-Tracking

**File:** `src/pipeline/05_backtrack.py`

---

## Overview

Stage 5 implements a **Lagrangian particle back-tracking** model using the **OceanParcels** framework (`parcels`). Starting from the spatial centroid of each confirmed marine debris cluster at the time of satellite detection, it releases an ensemble of virtual particles and integrates them **backward in time** through 3D surface ocean current fields (CMEMS) and 10 m wind fields (ERA5).

The endpoints of all particle trajectories form **probable marine debris origin regions** and source attribution candidates for Stage 6.

---

## Input

| Parameter | Type | Description |
|-----------|------|-------------|
| `scene_id` | `str` | Scene identifier |
| `detections_path` | `Path` | `detections_classified.geojson` from Stage 4 |
| `output_dir` | `Path` | Root output directory (default: `data/attribution`) |
| `config` | `dict` | Back-tracking parameters |
| `detection_date` | `str` (ISO 8601) | Satellite image acquisition date/time (start of backward integration) |
| `bbox` | `tuple` | Expanded geographic bounding box `[min_lon, min_lat, max_lon, max_lat]` |

---

## Output Structure

```
data/attribution/<SCENE_ID>/
├── backtrack_summary.json                # Complete source region centroids & bounding boxes for Stage 6
├── backtrack_<cluster_id>.geojson        # Particle trajectory polyline paths per cluster
├── backtrack_endpoints.geojson           # Backtracked particle endpoints
├── uncertainty_heatmap.geojson           # 95% spatial confidence convex hull polygon
├── ensemble_statistics.csv               # Hourly mean coordinates, std, covariance matrix, 95% ellipse, hull area
├── diagnostics.json                      # Transparent quality indicators (forcing completeness, smoothness, land ratio)
├── run_metadata.json                     # Git commit hash, tool versions, random seeds, kernel stack, dt, Kh
└── forcing_data/
    ├── ocean_currents.nc                 # CMEMS surface velocity fields (uo, vo)
    └── wind_data.nc                      # ERA5 10m wind velocity fields (u10, v10)
```

**Returns:** `List[Dict]` — list of source region dictionary structures.

---

## External Data Sources

### Ocean Currents — CMEMS

| Property | Value |
|----------|-------|
| API | Copernicus Marine Service (`copernicusmarine` CLI / Python API) |
| Dataset | `cmems_mod_glo_phy_anfc_merged-uv_PT1H-i` |
| Variables | `uo` (eastward surface velocity, m/s), `vo` (northward surface velocity, m/s) |
| Spatial / Temporal Resolution | ~8 km, 1-hour temporal resolution |
| Geographic Bounds | Detection bounding box + 5° spatial margin |

### Wind Data — ERA5 (ECMWF)

| Property | Value |
|----------|-------|
| API | Copernicus Climate Data Store (`cdsapi`) |
| Dataset | `reanalysis-era5-single-levels` |
| Variables | `10m_u_component_of_wind`, `10m_v_component_of_wind` |
| Resolution | ~31 km, 1-hour temporal resolution |

---

## Physical Kernels & Hydrodynamic Setup

The stage constructs an OceanParcels `FieldSet` combining ocean velocity vectors and windage components.

### 1. Advection Kernel (`AdvectionRK4`, `AdvectionRK45`, or `AdvectionEE`)
Solves 4th-order Runge-Kutta, adaptive RK45, or Euler-Forward backward particle advection:
$$\frac{d\vec{x}}{dt} = -\left( \vec{u}_{\text{ocean}} + \vec{u}_{\text{windage}} \right)$$

### 2. Direct Windage Kernel (`DirectWindageKernel`)
Applies object-specific aerodynamic wind drag directly to floating surface particles:
$$\vec{u}_{\text{windage}} = W_{\text{coeff}} \cdot \vec{U}_{10}$$

*Note: This represents aerodynamic surface drag, distinct from physical wave-driven Stokes drift ($S_{z=0}$).*

#### Debris-Specific Windage Lookup:
* `bottle`: `0.030` (3.0% windage coefficient)
* `fishing_net`: `0.015` (1.5% windage coefficient)
* `rope`: `0.020` (2.0% windage coefficient)
* `foam`: `0.050` (5.0% windage coefficient)
* `generic`: `0.030` (3.0% default windage coefficient)

### 3. Horizontal Brownian Diffusion (`BrownianDiffusion2D`)
Simulates sub-grid ocean turbulence and stochastic particle dispersal:
$$dx = \sqrt{2 \cdot K_h \cdot dt} \cdot R_x, \quad dy = \sqrt{2 \cdot K_h \cdot dt} \cdot R_y$$
Where $K_h = 1.5 \, m^2/s$ (configurable via `config.yaml`).

---

## Scientific Diagnostics & Quality Indicators

Instead of an opaque 0–100 score, Stage 5 produces transparent diagnostic metrics in `diagnostics.json`:

* `forcing_completeness_pct`: Percentage of simulation hours with valid forcing data.
* `ensemble_velocity_std_ms`: Standard deviation of particle velocity across ensemble.
* `trajectory_smoothness_index`: Turn angle variance indicator.
* `missing_timesteps_count`: Number of missing forcing steps.
* `land_intersection_ratio_pct`: Percentage of particles intersecting coastal boundaries.

---

## Visualization Tools

Stage 5 auto-generates visual outputs when run via the standalone verification tools:
1. **Interactive Leaflet HTML Map (`backtrack_map.html`)**: Interactive map with CartoDB Light/Dark/OSM basemaps, release marker, origin centroid, 95% spatial confidence boundary, and 100 polyline trajectory paths (generated via `verification/generate_visualizations.py`).
2. **Matplotlib Dashboard (`backtrack_dashboard.png`)**: 4-panel publication plot displaying spatial trajectories, cumulative drift distance over time, spread radius growth ($km$), and velocity distributions.

*(Note: During full pipeline execution, the interactive HTML map is generated downstream in Stage 7.)*

---

## Config Keys Used (`src/config/config.yaml`)

```yaml
backtracking:
  days:                 30    # Integration duration (days)
  n_particles:          100   # Ensemble size per cluster
  time_step_hours:       1    # Integration timestep
  integrator:          "RK4" # RK4, RK45, or Euler
  horizontal_diffusion:
    enabled: true
    Kh: 1.5                    # Diffusion coefficient in m^2/s
  stokes_drift:
    enabled: false
    source: "CMEMS_WAVE"
  plastic_windage:
    bottle: 0.030
    fishing_net: 0.015
    rope: 0.020
    foam: 0.050
    generic: 0.030
  uncertainty:
    compute_ellipse: true
    confidence_level: 0.95
  diagnostics:
    export_indicators: true
  dbscan_eps_degrees:  0.5
  dbscan_min_samples:    5
```

---

## Verification & Benchmark Suite

Verification scripts are located in `verification/`:
```bash
# Single benchmark run
python verification/verify_backtrack.py --output_dir data/benchmarks/benchmark_run_2 --lat 41.20218 --lon 2.28279 --start_str "2022-03-12T18:11:00+01:00" --end_str "2022-03-15T18:11:00+01:00"

# Multi-scale Monte Carlo analysis (100, 500, 1000 particles)
python verification/verify_backtrack.py --output_dir data/benchmarks/benchmark_run_2 --lat 41.20218 --lon 2.28279 --start_str "2022-03-12T18:11:00+01:00" --end_str "2022-03-15T18:11:00+01:00" --monte-carlo

# Render dynamic visualizations
python verification/generate_visualizations.py data/benchmarks/benchmark_run_2
```

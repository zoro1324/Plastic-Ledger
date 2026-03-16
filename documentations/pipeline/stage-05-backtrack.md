# Stage 5: Hydrodynamic Back-Tracking

**File:** `pipeline/05_backtrack.py`

---

## Overview

Stage 5 implements a **Lagrangian particle back-tracking** model. Starting from the centroid of each confirmed debris cluster at the time of satellite detection, it releases virtual particles and integrates them **backward in time** through ocean current and wind fields. The endpoints of all particle trajectories cluster into **probable source regions**.

---

## Input

| Parameter | Type | Description |
|-----------|------|-------------|
| `scene_id` | `str` | Scene identifier |
| `detections_path` | `Path` | `detections_classified.geojson` from Stage 4 |
| `output_dir` | `Path` | Root output directory (default: `data/attribution`) |
| `config` | `dict` | Back-tracking parameters |
| `detection_date` | `str` (ISO 8601) | Date of the satellite image (start of backward integration) |
| `bbox` | `tuple` | Bounding box for current/wind data download |

---

## Output

```
data/attribution/<SCENE_ID>/
├── backtrack_summary.json              # Source region list for Stage 6
├── backtrack_<cluster_id>.geojson      # Trajectory lines per cluster
└── forcing_data/
    ├── ocean_currents.nc               # CMEMS current data (NetCDF)
    └── wind_data.nc                    # ERA5 wind data (NetCDF)
```

**Returns:** `List[Dict]` — list of source region dicts.

---

## External Data Sources

### Ocean Currents — CMEMS

| Property | Value |
|----------|-------|
| API | Copernicus Marine Service (copernicusmarine Python package) |
| Dataset | `cmems_mod_glo_phy_anfc_merged-uv_PT1H-i` |
| Variables | `uo` (eastward velocity, m/s), `vo` (northward velocity, m/s) |
| Resolution | ~8 km, hourly |
| Domain | Expanded bounding box + 5° padding |

```python
cm.subset(
    dataset_id="cmems_mod_glo_phy_anfc_merged-uv_PT1H-i",
    variables=["uo", "vo"],
    minimum_longitude=bbox[0], maximum_longitude=bbox[2],
    minimum_latitude=bbox[1],  maximum_latitude=bbox[3],
    start_datetime=date_start, end_datetime=date_end,
    ...
)
```

### Wind Data — ERA5 (ECMWF)

| Property | Value |
|----------|-------|
| API | CDS API (`cdsapi` Python package) |
| Dataset | `reanalysis-era5-single-levels` |
| Variables | `10m_u_component_of_wind`, `10m_v_component_of_wind` |
| Resolution | ~31 km, hourly |

```python
c.retrieve("reanalysis-era5-single-levels", {
    "product_type": "reanalysis",
    "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
    "area": [N, W, S, E],
    "date": f"{date_start}/{date_end}",
    "time": ["00:00", "01:00", ..., "23:00"],
    "format": "netcdf",
})
```

### Fallback (Synthetic Currents)

If either API is unavailable/unauthenticated, a seeded random velocity is generated:
```python
rng = np.random.default_rng(int(abs(lon * 1000 + lat * 1000)))
u = rng.normal(0.02, 0.01)   # ~0.02 m/s eastward
v = rng.normal(0.01, 0.01)   # ~0.01 m/s northward
```

---

## Combined Velocity Model

The total velocity at any point combines ocean current and wind-driven drift (Stokes drift approximation):

```
u_total = w_ocean × u_ocean + w_wind × u_wind
v_total = w_ocean × v_ocean + w_wind × v_wind
```

Default weights (from config):
```
w_ocean = 0.97   # 97% ocean current
w_wind  = 0.03   # 3% wind-driven drift
```

---

## Unit Conversion

Ocean current velocities are in m/s. They must be converted to degrees/hour for geographic particle movement:

```
deg_per_m_lat = 1 / 111,000           # 1° lat ≈ 111 km

deg_per_m_lon = 1 / (111,000 × cos(lat))   # accounts for meridional compression

dx_dt = u_total × deg_per_m_lon × 3600   # degrees per hour
dy_dt = v_total × deg_per_m_lat × 3600
```

---

## RK4 Backward Integration

Each particle is integrated backward in time using the **4th-order Runge-Kutta** scheme:

```
# Time step: dt (hours)
# Backward: negate velocities

k1x, k1y = -vel(lon,              lat,              t)
k2x, k2y = -vel(lon + k1x×dt/2,  lat + k1y×dt/2,  t - dt/2)
k3x, k3y = -vel(lon + k2x×dt/2,  lat + k2y×dt/2,  t - dt/2)
k4x, k4y = -vel(lon + k3x×dt,    lat + k3y×dt,    t - dt)

dlon = (k1x + 2×k2x + 2×k3x + k4x) / 6 × dt
dlat = (k1y + 2×k2y + 2×k3y + k4y) / 6 × dt

new_lon = lon + dlon
new_lat = lat + dlat
```

**Boundary conditions:**
```python
new_lat = clip(new_lat, -85, 85)           # Polar cap
new_lon = ((new_lon + 180) % 360) - 180   # Longitude wrap-around
```

---

## Particle Release

For each debris cluster, `n_particles` (default: 50) particles are released with small random offsets around the cluster centroid:

```python
rng = np.random.default_rng(cluster_id + 42)
p_lon = centroid.x + rng.normal(0, 0.01)   # ±0.01° ≈ ±1 km
p_lat = centroid.y + rng.normal(0, 0.01)
```

Each particle is integrated backward for `bt_days × 24` hours at `dt_hours = 1.0` hour steps.

---

## Endpoint Clustering (DBSCAN)

After back-tracking, the final positions (oldest points in each trajectory) are clustered to identify **source regions**:

```python
DBSCAN(
    eps=eps_degrees,       # 0.5° ≈ 55 km
    min_samples=5          # Minimum density for a cluster
).fit(endpoints)
```

Each DBSCAN cluster becomes a **source region**:

```json
{
  "source_centroid": [lon, lat],
  "source_bbox":     [lon_min, lat_min, lon_max, lat_max],
  "source_probability": 0.72,   // fraction of particles ending here
  "n_particles": 36,
  "cluster_id": 0,
  "days_to_source": 30
}
```

`source_probability = cluster_particle_count / total_particles`

---

## Processing Steps

```
1. Load classified detections, filter out false positives
       ↓
2. Determine detection datetime and back-track date range
       ↓
3. Download ocean currents (CMEMS) for detection − bt_days to detection
       ↓
4. Download wind data (ERA5) for the same period
       ↓
5. For each confirmed debris cluster:
     a. Release n_particles (default 50) with ±1 km random offsets
     b. For each particle:
          - Run RK4 backward integration for bt_days × 24 hours
          - Record trajectory (lon, lat, time) at 1-hour steps
     c. Extract trajectory endpoint (oldest time step)
     d. DBSCAN-cluster endpoints → source regions
     e. Save backtrack_<id>.geojson with trajectory lines
       ↓
6. Save backtrack_summary.json
```

---

## Config Keys Used

```yaml
backtracking:
  days:                 30    # How far back to integrate (days)
  n_particles:          50    # Particles per debris cluster
  time_step_hours:       1    # RK4 time step
  ocean_wind_ratio: [0.97, 0.03]  # [ocean weight, wind weight]
  dbscan_eps_degrees:  0.5   # DBSCAN spatial epsilon
  dbscan_min_samples:    5   # DBSCAN minimum cluster density
```

---

## Caching

If `backtrack_summary.json` already exists, the stage is skipped and the existing data is returned.

---

## CLI Usage

```bash
python -m pipeline.05_backtrack \
    --scene_id       S2A_MSIL2A_20240115... \
    --detections     data/detections/S2A.../detections_classified.geojson \
    --output_dir     data/attribution \
    --detection_date 2024-01-15 \
    --bbox           "80.0,8.0,82.0,10.0" \
    --config         config/config.yaml
```

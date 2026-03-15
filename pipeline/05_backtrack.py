"""
Plastic-Ledger — Stage 5: Hydrodynamic Back-Tracking
======================================================
Implements a Lagrangian particle back-tracking model that traces detected
debris back to likely source points using ocean current and wind data.

Usage (standalone):
    python -m pipeline.05_backtrack \\
        --scene_id SCENE_ID \\
        --detections data/detections/SCENE_ID/detections_classified.geojson \\
        --output_dir data/attribution

Dependencies: copernicusmarine, cdsapi, numpy, scipy, geopandas, sklearn, xarray
"""

import argparse
import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point, MultiPoint, box
from sklearn.cluster import DBSCAN

from pipeline.utils.logging_utils import get_logger
from pipeline.utils.geo_utils import expand_bbox, retry_request
from pipeline.utils.cache_utils import load_config, stage_output_exists

logger = get_logger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ─────────────────────────────────────────────
# OCEAN CURRENT DATA
# ─────────────────────────────────────────────
def download_ocean_currents(
    bbox: Tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    output_dir: Path,
) -> Optional[Path]:
    """Download CMEMS global ocean surface current data.

    Args:
        bbox: ``(lon_min, lat_min, lon_max, lat_max)``.
        date_start: Start date ISO string.
        date_end: End date ISO string.
        output_dir: Directory to save the NetCDF file.

    Returns:
        Path to the downloaded NetCDF, or ``None`` if download fails.

    Raises:
        ImportError: If ``copernicusmarine`` is not installed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ocean_currents.nc"

    if out_path.exists():
        logger.info("Ocean currents already cached: %s", out_path)
        return out_path

    try:
        import copernicusmarine as cm

        @retry_request
        def _download():
            cm.subset(
                dataset_id="cmems_mod_glo_phy_anfc_merged-uv_PT1H-i",
                variables=["uo", "vo"],
                minimum_longitude=bbox[0],
                minimum_latitude=bbox[1],
                maximum_longitude=bbox[2],
                maximum_latitude=bbox[3],
                start_datetime=date_start,
                end_datetime=date_end,
                output_directory=str(output_dir),
                output_filename="ocean_currents.nc",
                force_download=True,
            )

        _download()
        logger.info("Downloaded ocean currents to %s", out_path)
        return out_path

    except ImportError:
        logger.warning(
            "copernicusmarine not installed — using synthetic currents"
        )
        return None
    except Exception as exc:
        logger.warning("CMEMS download failed: %s — using synthetic currents", exc)
        return None


# ─────────────────────────────────────────────
# WIND DATA
# ─────────────────────────────────────────────
def download_wind_data(
    bbox: Tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    output_dir: Path,
) -> Optional[Path]:
    """Download ERA5 10m wind components.

    Args:
        bbox: ``(lon_min, lat_min, lon_max, lat_max)``.
        date_start: Start date ISO string.
        date_end: End date ISO string.
        output_dir: Directory to save the NetCDF file.

    Returns:
        Path to the downloaded NetCDF, or ``None`` if download fails.

    Raises:
        ImportError: If ``cdsapi`` is not installed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "wind_data.nc"

    if out_path.exists():
        logger.info("Wind data already cached: %s", out_path)
        return out_path

    try:
        import cdsapi

        @retry_request
        def _download():
            c = cdsapi.Client()
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": [
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                    ],
                    "area": [bbox[3], bbox[0], bbox[1], bbox[2]],  # N, W, S, E
                    "date": f"{date_start}/{date_end}",
                    "time": [f"{h:02d}:00" for h in range(24)],
                    "format": "netcdf",
                },
                str(out_path),
            )

        _download()
        logger.info("Downloaded wind data to %s", out_path)
        return out_path

    except ImportError:
        logger.warning("cdsapi not installed — using synthetic wind data")
        return None
    except Exception as exc:
        logger.warning("ERA5 download failed: %s — using synthetic wind", exc)
        return None


# ─────────────────────────────────────────────
# CURRENT/WIND INTERPOLATION
# ─────────────────────────────────────────────
def _load_velocity_field(nc_path: Optional[Path]) -> Optional[Any]:
    """Load a NetCDF velocity field as an xarray Dataset.

    Args:
        nc_path: Path to the NetCDF file.

    Returns:
        :class:`xarray.Dataset` or ``None``.
    """
    if nc_path is None or not nc_path.exists():
        return None
    try:
        import xarray as xr
        return xr.open_dataset(nc_path)
    except Exception as exc:
        logger.warning("Failed to load %s: %s", nc_path, exc)
        return None


def _interpolate_velocity(
    ds: Optional[Any],
    lon: float,
    lat: float,
    time_dt: datetime,
    u_var: str = "uo",
    v_var: str = "vo",
) -> Tuple[float, float]:
    """Interpolate velocity at a point from gridded data.

    Args:
        ds: xarray Dataset with velocity fields.
        lon: Longitude.
        lat: Latitude.
        time_dt: Datetime for temporal interpolation.
        u_var: Name of eastward velocity variable.
        v_var: Name of northward velocity variable.

    Returns:
        ``(u, v)`` velocity in m/s.
    """
    if ds is None:
        # Synthetic fallback: small random current
        rng = np.random.default_rng(int(abs(lon * 1000 + lat * 1000)))
        return float(rng.normal(0.02, 0.01)), float(rng.normal(0.01, 0.01))

    try:
        u = float(
            ds[u_var].sel(
                longitude=lon, latitude=lat, time=time_dt,
                method="nearest",
            ).values
        )
        v = float(
            ds[v_var].sel(
                longitude=lon, latitude=lat, time=time_dt,
                method="nearest",
            ).values
        )
        if np.isnan(u):
            u = 0.0
        if np.isnan(v):
            v = 0.0
        return u, v
    except Exception:
        return 0.0, 0.0


# ─────────────────────────────────────────────
# RK4 INTEGRATION
# ─────────────────────────────────────────────
def _velocity_at(
    lon: float,
    lat: float,
    time_dt: datetime,
    ocean_ds: Any,
    wind_ds: Any,
    ocean_wind_ratio: Tuple[float, float] = (0.97, 0.03),
) -> Tuple[float, float]:
    """Compute total velocity at a point (ocean + wind drift).

    Args:
        lon: Longitude.
        lat: Latitude.
        time_dt: Current time.
        ocean_ds: Ocean current dataset.
        wind_ds: Wind dataset.
        ocean_wind_ratio: ``(ocean_weight, wind_weight)``.

    Returns:
        ``(dx_dt, dy_dt)`` in degrees/hour (approximate).
    """
    u_ocean, v_ocean = _interpolate_velocity(ocean_ds, lon, lat, time_dt)
    u_wind, v_wind = _interpolate_velocity(
        wind_ds, lon, lat, time_dt, u_var="u10", v_var="v10",
    )

    # Combine with Stokes drift approximation
    u = ocean_wind_ratio[0] * u_ocean + ocean_wind_ratio[1] * u_wind
    v = ocean_wind_ratio[0] * v_ocean + ocean_wind_ratio[1] * v_wind

    # Convert m/s to degrees/hour (approximate)
    # 1 degree latitude ≈ 111,000 m
    # 1 degree longitude ≈ 111,000 * cos(lat) m
    deg_per_m_lat = 1.0 / 111_000.0
    deg_per_m_lon = 1.0 / (111_000.0 * np.cos(np.radians(lat)) + 1e-10)

    dx_dt = u * deg_per_m_lon * 3600  # degrees/hour
    dy_dt = v * deg_per_m_lat * 3600

    return dx_dt, dy_dt


def backtrack_particle(
    start_lon: float,
    start_lat: float,
    start_time: datetime,
    ocean_ds: Any,
    wind_ds: Any,
    hours: int = 720,
    dt_hours: float = 1.0,
    ocean_wind_ratio: Tuple[float, float] = (0.97, 0.03),
) -> List[Tuple[float, float, datetime]]:
    """Back-track a single particle using RK4 integration.

    Args:
        start_lon: Starting longitude.
        start_lat: Starting latitude.
        start_time: Detection time.
        ocean_ds: Ocean current dataset.
        wind_ds: Wind dataset.
        hours: Total hours to integrate backward.
        dt_hours: Time step in hours.
        ocean_wind_ratio: ``(ocean_weight, wind_weight)``.

    Returns:
        List of ``(lon, lat, time)`` trajectory points (oldest first).
    """
    trajectory = [(start_lon, start_lat, start_time)]
    lon, lat = start_lon, start_lat
    t = start_time

    # Simple RK4 land check: keep particles in water (lat ∈ [-85, 85])
    n_steps = int(hours / dt_hours)

    for _ in range(n_steps):
        t_prev = t
        t = t - timedelta(hours=dt_hours)

        # RK4 (backward)
        def vel(lo, la, ti):
            dx, dy = _velocity_at(lo, la, ti, ocean_ds, wind_ds, ocean_wind_ratio)
            return -dx, -dy  # negative = backward in time

        k1x, k1y = vel(lon, lat, t_prev)
        t_mid = t_prev - timedelta(hours=dt_hours / 2)
        k2x, k2y = vel(lon + k1x * dt_hours / 2, lat + k1y * dt_hours / 2, t_mid)
        k3x, k3y = vel(lon + k2x * dt_hours / 2, lat + k2y * dt_hours / 2, t_mid)
        k4x, k4y = vel(lon + k3x * dt_hours, lat + k3y * dt_hours, t)

        dlon = (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0 * dt_hours
        dlat = (k1y + 2 * k2y + 2 * k3y + k4y) / 6.0 * dt_hours

        new_lon = lon + dlon
        new_lat = lat + dlat

        # Simple land check: clamp to ocean (very rough)
        new_lat = np.clip(new_lat, -85, 85)
        new_lon = ((new_lon + 180) % 360) - 180  # wrap longitude

        lon, lat = new_lon, new_lat
        trajectory.append((lon, lat, t))

    trajectory.reverse()  # oldest first
    return trajectory


# ─────────────────────────────────────────────
# ENDPOINT CLUSTERING
# ─────────────────────────────────────────────
def cluster_endpoints(
    endpoints: List[Tuple[float, float]],
    eps_degrees: float = 0.5,
    min_samples: int = 5,
) -> List[Dict[str, Any]]:
    """Cluster back-tracked particle endpoints using DBSCAN.

    Args:
        endpoints: List of ``(lon, lat)`` endpoint coordinates.
        eps_degrees: DBSCAN epsilon in degrees.
        min_samples: DBSCAN minimum samples per cluster.

    Returns:
        List of source region dicts with keys ``source_centroid``,
        ``source_bbox``, ``source_probability``, ``n_particles``.
    """
    if len(endpoints) < min_samples:
        # Not enough points for DBSCAN
        if endpoints:
            centroid = np.mean(endpoints, axis=0)
            return [{
                "source_centroid": (float(centroid[0]), float(centroid[1])),
                "source_bbox": (
                    float(min(e[0] for e in endpoints)),
                    float(min(e[1] for e in endpoints)),
                    float(max(e[0] for e in endpoints)),
                    float(max(e[1] for e in endpoints)),
                ),
                "source_probability": 1.0,
                "n_particles": len(endpoints),
            }]
        return []

    coords = np.array(endpoints)
    db = DBSCAN(eps=eps_degrees, min_samples=min_samples).fit(coords)

    labels = db.labels_
    unique_labels = set(labels) - {-1}
    n_total = len(endpoints)

    sources = []
    for label in sorted(unique_labels):
        mask = labels == label
        cluster_pts = coords[mask]
        centroid = cluster_pts.mean(axis=0)

        sources.append({
            "source_centroid": (float(centroid[0]), float(centroid[1])),
            "source_bbox": (
                float(cluster_pts[:, 0].min()),
                float(cluster_pts[:, 1].min()),
                float(cluster_pts[:, 0].max()),
                float(cluster_pts[:, 1].max()),
            ),
            "source_probability": float(mask.sum()) / n_total,
            "n_particles": int(mask.sum()),
        })

    # Sort by probability descending
    sources.sort(key=lambda x: x["source_probability"], reverse=True)
    return sources


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────
def run(
    scene_id: str,
    detections_path: Union[str, Path],
    output_dir: Union[str, Path] = "data/attribution",
    config: Optional[Dict] = None,
    detection_date: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> List[Dict[str, Any]]:
    """Run back-tracking for all confirmed debris clusters.

    Args:
        scene_id: Scene identifier.
        detections_path: Path to classified detections GeoJSON.
        output_dir: Root output directory.
        config: Optional config dict.
        detection_date: ISO date string for the detection.
        bbox: Original search bounding box.

    Returns:
        List of source region dicts per debris cluster.

    Raises:
        FileNotFoundError: If *detections_path* does not exist.
    """
    detections_path = Path(detections_path)
    out_dir = Path(output_dir) / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Settings from config
    bt_days = 30
    n_particles = 50
    dt_hours = 1.0
    ocean_wind_ratio = (0.97, 0.03)
    eps_degrees = 0.5
    min_samples = 5

    if config:
        bt_cfg = config.get("backtracking", {})
        bt_days = bt_cfg.get("days", 30)
        n_particles = bt_cfg.get("n_particles", 50)
        dt_hours = bt_cfg.get("time_step_hours", 1.0)
        ratio = bt_cfg.get("ocean_wind_ratio", [0.97, 0.03])
        ocean_wind_ratio = (ratio[0], ratio[1])
        eps_degrees = bt_cfg.get("dbscan_eps_degrees", 0.5)
        min_samples = bt_cfg.get("dbscan_min_samples", 5)

    # Check cache
    if stage_output_exists(out_dir, ["backtrack_summary.json"]):
        with open(out_dir / "backtrack_summary.json") as fh:
            return json.load(fh)

    # Load detections
    gdf = gpd.read_file(detections_path)

    # Filter out false positives
    if "is_false_positive" in gdf.columns:
        gdf = gdf[gdf["is_false_positive"] != True].reset_index(drop=True)

    if len(gdf) == 0:
        logger.info("No confirmed debris clusters — skipping backtracking")
        summary = []
        with open(out_dir / "backtrack_summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        return summary

    # Determine dates
    if detection_date:
        det_dt = datetime.fromisoformat(detection_date.replace("Z", "+00:00"))
    else:
        # Try from detections
        if "detection_date" in gdf.columns and gdf["detection_date"].iloc[0]:
            try:
                det_dt = datetime.fromisoformat(
                    str(gdf["detection_date"].iloc[0]).replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                det_dt = datetime.now()
        else:
            det_dt = datetime.now()

    bt_start = (det_dt - timedelta(days=bt_days)).isoformat()
    bt_end = det_dt.isoformat()

    # Determine bbox for data download
    if bbox is None:
        all_lons = gdf.geometry.centroid.x.tolist()
        all_lats = gdf.geometry.centroid.y.tolist()
        bbox = (min(all_lons), min(all_lats), max(all_lons), max(all_lats))

    expanded_bbox = expand_bbox(bbox, 5.0)

    # Download current and wind data
    data_dir = out_dir / "forcing_data"
    data_dir.mkdir(exist_ok=True)

    logger.info("Downloading ocean current data")
    ocean_path = download_ocean_currents(expanded_bbox, bt_start, bt_end, data_dir)
    logger.info("Downloading wind data")
    wind_path = download_wind_data(expanded_bbox, bt_start, bt_end, data_dir)

    ocean_ds = _load_velocity_field(ocean_path)
    wind_ds = _load_velocity_field(wind_path)

    # Run back-tracking for each cluster
    all_sources = []
    total_hours = bt_days * 24

    for idx, row in gdf.iterrows():
        cluster_id = row.get("cluster_id", idx)
        centroid = row.geometry.centroid

        logger.info(
            "Back-tracking cluster %s (%d particles, %d days)",
            cluster_id, n_particles, bt_days,
        )

        # Release particles with small random offsets
        rng = np.random.default_rng(int(cluster_id) + 42)
        endpoints = []
        all_trajectories = []

        for p in range(n_particles):
            # Add small random offset (±0.01 degrees ≈ ±1km)
            p_lon = centroid.x + rng.normal(0, 0.01)
            p_lat = centroid.y + rng.normal(0, 0.01)

            traj = backtrack_particle(
                p_lon, p_lat, det_dt,
                ocean_ds, wind_ds,
                hours=total_hours,
                dt_hours=dt_hours,
                ocean_wind_ratio=ocean_wind_ratio,
            )

            endpoints.append((traj[0][0], traj[0][1]))  # oldest point
            all_trajectories.append(traj)

        # Cluster endpoints
        sources = cluster_endpoints(endpoints, eps_degrees, min_samples)

        # Compute days_to_source for each source
        for src in sources:
            src["cluster_id"] = int(cluster_id)
            src["days_to_source"] = float(bt_days)

        all_sources.extend(sources)

        # Save trajectory GeoJSON per cluster
        traj_features = []
        for traj in all_trajectories:
            coords = [(pt[0], pt[1]) for pt in traj]
            if len(coords) >= 2:
                traj_features.append(LineString(coords))

        if traj_features:
            traj_gdf = gpd.GeoDataFrame(
                {"geometry": traj_features, "cluster_id": [int(cluster_id)] * len(traj_features)},
                crs="EPSG:4326",
            )
            traj_path = out_dir / f"backtrack_{cluster_id}.geojson"
            traj_gdf.to_file(traj_path, driver="GeoJSON")

    # Save summary
    with open(out_dir / "backtrack_summary.json", "w") as fh:
        json.dump(all_sources, fh, indent=2, default=str)

    # Clean up datasets
    if ocean_ds is not None:
        ocean_ds.close()
    if wind_ds is not None:
        wind_ds.close()

    logger.info(
        "[bold green]Stage 5 complete[/] — %d source regions from %d clusters",
        len(all_sources), len(gdf),
    )
    return all_sources


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    """CLI entrypoint for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Stage 5: Hydrodynamic back-tracking of debris clusters",
    )
    parser.add_argument("--scene_id", type=str, required=True)
    parser.add_argument("--detections", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/attribution")
    parser.add_argument("--detection_date", type=str, default=None)
    parser.add_argument("--bbox", type=str, default=None)
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    bbox = None
    if args.bbox:
        bbox = tuple(float(x) for x in args.bbox.split(","))

    sources = run(
        scene_id=args.scene_id,
        detections_path=args.detections,
        output_dir=args.output_dir,
        config=config,
        detection_date=args.detection_date,
        bbox=bbox,
    )
    print(f"\nFound {len(sources)} source regions")


if __name__ == "__main__":
    main()

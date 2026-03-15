"""
Plastic-Ledger — Stage 1: Satellite Data Ingestion
====================================================
Downloads Sentinel-2 L2A imagery from the Copernicus Data Space Ecosystem
using their STAC API.

Usage (standalone):
    python -m pipeline.01_ingest \\
        --bbox "80.0,8.0,82.0,10.0" \\
        --start_date 2024-01-01 \\
        --end_date 2024-01-31 \\
        --output_dir data/raw

Dependencies: pystac-client, requests, rasterio, shapely
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
import rasterio
from rasterio.transform import from_bounds
from pystac_client import Client
from shapely.geometry import box

from pipeline.utils.logging_utils import get_logger
from pipeline.utils.geo_utils import retry_request
from pipeline.utils.cache_utils import load_config, stage_output_exists

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
STAC_ENDPOINT = "https://catalogue.dataspace.copernicus.eu/stac"
REQUIRED_BANDS = ["B02", "B03", "B04", "B05", "B08", "B8A", "B11", "B12"]
COLLECTION = "sentinel-2-l2a"


# ─────────────────────────────────────────────
# STAC SEARCH
# ─────────────────────────────────────────────
def search_scenes(
    bbox: Tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    cloud_cover_max: int = 20,
    max_items: int = 50,
) -> List[Dict[str, Any]]:
    """Search the Copernicus STAC catalogue for Sentinel-2 L2A scenes.

    Args:
        bbox: ``(lon_min, lat_min, lon_max, lat_max)``.
        date_start: ISO date string (e.g. ``2024-01-01``).
        date_end: ISO date string.
        cloud_cover_max: Maximum cloud cover percentage (0–100).
        max_items: Cap on number of items returned.

    Returns:
        List of STAC item dicts with id, datetime, cloud_cover, assets, bbox.

    Raises:
        ConnectionError: If the STAC endpoint is unreachable.
        RuntimeError: If no scenes are found matching the criteria.
    """
    logger.info(
        "Searching STAC: bbox=%s, dates=%s to %s, cloud≤%d%%",
        bbox, date_start, date_end, cloud_cover_max,
    )

    catalog = Client.open(STAC_ENDPOINT)
    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=f"{date_start}/{date_end}",
        query={"eo:cloud_cover": {"lt": cloud_cover_max}},
        max_items=max_items,
    )

    items = list(search.items())
    if not items:
        raise RuntimeError(
            f"No Sentinel-2 L2A scenes found for bbox={bbox}, "
            f"dates={date_start}–{date_end}, cloud<{cloud_cover_max}%. "
            "Try expanding the date range or increasing cloud_cover_max."
        )

    logger.info("Found [bold cyan]%d[/] scenes", len(items))

    results = []
    for item in items:
        results.append({
            "id": item.id,
            "datetime": str(item.datetime),
            "cloud_cover": item.properties.get("eo:cloud_cover", None),
            "bbox": item.bbox,
            "assets": {
                k: v.extra_fields.get("alternate", {}).get("https", {}).get("href", v.href)
                for k, v in item.assets.items()
            },
            "geometry": item.geometry,
        })

    # Sort by cloud cover ascending
    results.sort(key=lambda x: x.get("cloud_cover", 100))
    return results


# ─────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────
def _download_file(url: str, dest: Path, session: requests.Session) -> Path:
    """Download a single file with retry logic.

    Args:
        url: Direct download URL.
        dest: Destination file path.
        session: Authenticated requests session.

    Returns:
        Path to the downloaded file.

    Raises:
        requests.HTTPError: If download fails after retries.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("  Already downloaded: %s", dest.name)
        return dest

    @retry_request
    def _do_download():
        resp = session.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                fh.write(chunk)

    _do_download()
    logger.info("  Downloaded: %s", dest.name)
    return dest


def download_scene(
    scene: Dict[str, Any],
    output_dir: Path,
    bands: List[str] = None,
    session: Optional[requests.Session] = None,
) -> Tuple[Path, Dict[str, str]]:
    """Download band assets for a single STAC scene.

    Args:
        scene: Scene dict from :func:`search_scenes`.
        output_dir: Base output directory (scene subfolder is created).
        bands: Band names to download.  Defaults to :data:`REQUIRED_BANDS`.
        session: Optional pre-authenticated :class:`requests.Session`.

    Returns:
        Tuple of (scene directory path, dict mapping band name → file path).

    Raises:
        KeyError: If a required band is missing from the scene assets.
    """
    bands = bands or REQUIRED_BANDS
    scene_id = scene["id"]
    scene_dir = output_dir / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    if session is None:
        session = requests.Session()

    logger.info("Downloading scene [bold]%s[/] (%d bands)", scene_id, len(bands))

    band_paths = {}
    for band_name in bands:
        # STAC assets may be keyed as "B02_10m", "B02", etc.
        asset_key = None
        for key in scene["assets"]:
            if band_name.lower() in key.lower():
                asset_key = key
                break

        if asset_key is None:
            logger.warning(
                "Band %s not found in scene %s assets. Available: %s",
                band_name, scene_id, list(scene["assets"].keys()),
            )
            continue

        url = scene["assets"][asset_key]
        dest = scene_dir / f"{band_name}.tif"

        try:
            _download_file(url, dest, session)
            band_paths[band_name] = str(dest)
        except Exception as exc:
            logger.error("Failed to download %s: %s", band_name, exc)

    # Save scene metadata
    meta_path = scene_dir / "metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(scene, fh, indent=2, default=str)

    return scene_dir, band_paths


# ─────────────────────────────────────────────
# AUTHENTICATION
# ─────────────────────────────────────────────
def get_copernicus_session(username: str, password: str) -> requests.Session:
    """Authenticate with Copernicus Data Space and return an authorized session.

    Args:
        username: Copernicus account email.
        password: Copernicus account password.

    Returns:
        :class:`requests.Session` with bearer token set.

    Raises:
        RuntimeError: If authentication fails.
    """
    token_url = (
        "https://identity.dataspace.copernicus.eu/auth/realms/"
        "CDSE/protocol/openid-connect/token"
    )

    @retry_request
    def _get_token():
        resp = requests.post(
            token_url,
            data={
                "grant_type": "password",
                "username": username,
                "password": password,
                "client_id": "cdse-public",
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["access_token"]

    token = _get_token()

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    logger.info("Authenticated with Copernicus Data Space")
    return session


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────
def run(
    bbox: Tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    cloud_cover_max: int = 20,
    output_dir: Union[str, Path] = "data/raw",
    config: Optional[Dict] = None,
) -> Tuple[List[Path], List[Dict]]:
    """Run the full ingestion stage.

    Args:
        bbox: ``(lon_min, lat_min, lon_max, lat_max)``.
        date_start: Start date ISO string.
        date_end: End date ISO string.
        cloud_cover_max: Max cloud cover %.
        output_dir: Root output directory for raw scenes.
        config: Optional config dict (for API credentials).

    Returns:
        Tuple of (list of scene directory paths, list of scene metadata dicts).

    Raises:
        RuntimeError: If no scenes found or all downloads fail.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Search
    scenes = search_scenes(bbox, date_start, date_end, cloud_cover_max)

    # Authenticate (if credentials available)
    session = None
    if config and config.get("apis", {}).get("copernicus_username"):
        try:
            session = get_copernicus_session(
                config["apis"]["copernicus_username"],
                config["apis"]["copernicus_password"],
            )
        except Exception as exc:
            logger.warning("Auth failed (%s) — downloading without auth", exc)

    # Download each scene
    scene_dirs = []
    scene_metas = []
    for scene in scenes:
        try:
            scene_dir, band_paths = download_scene(
                scene, output_dir, session=session,
            )
            scene_dirs.append(scene_dir)
            scene_metas.append(scene)
            logger.info(
                "Scene %s: %d bands downloaded", scene["id"], len(band_paths),
            )
        except Exception as exc:
            logger.error("Skipping scene %s: %s", scene["id"], exc)

    if not scene_dirs:
        raise RuntimeError("All scene downloads failed.")

    # Save run metadata
    run_meta = {
        "bbox": list(bbox),
        "date_start": date_start,
        "date_end": date_end,
        "cloud_cover_max": cloud_cover_max,
        "scenes": [s["id"] for s in scene_metas],
        "scene_paths": [str(d) for d in scene_dirs],
    }
    with open(output_dir / "ingest_metadata.json", "w") as fh:
        json.dump(run_meta, fh, indent=2)

    logger.info(
        "[bold green]Stage 1 complete[/] — %d scenes in %s",
        len(scene_dirs), output_dir,
    )
    return scene_dirs, scene_metas


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    """CLI entrypoint for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Stage 1: Download Sentinel-2 L2A scenes from Copernicus STAC",
    )
    parser.add_argument(
        "--bbox", type=str, required=True,
        help="Bounding box as 'lon_min,lat_min,lon_max,lat_max'",
    )
    parser.add_argument("--start_date", type=str, required=True)
    parser.add_argument("--end_date", type=str, required=True)
    parser.add_argument("--cloud_cover", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="data/raw")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    bbox = tuple(float(x) for x in args.bbox.split(","))
    assert len(bbox) == 4, "bbox must have exactly 4 values"

    config = load_config(args.config)
    scene_dirs, scene_metas = run(
        bbox=bbox,
        date_start=args.start_date,
        date_end=args.end_date,
        cloud_cover_max=args.cloud_cover,
        output_dir=args.output_dir,
        config=config,
    )
    print(f"\nDownloaded {len(scene_dirs)} scenes to {args.output_dir}")


if __name__ == "__main__":
    main()

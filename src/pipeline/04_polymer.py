"""
Plastic-Ledger — Stage 4: Polymer Type Classification
========================================================
Classifies detected debris clusters by polymer type using SWIR/NIR spectral
ratios and a rule-based decision tree.

Usage (standalone):
    python -m pipeline.04_polymer \\
        --scene_id SCENE_ID \\
        --detections data/detections/SCENE_ID/detections.geojson \\
        --processed_dir data/processed/SCENE_ID

Dependencies: geopandas, rasterio, numpy, shapely
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import Affine, rowcol
from shapely.geometry import mapping

from pipeline.utils.logging_utils import get_logger
from pipeline.utils.cache_utils import load_config, stage_output_exists

logger = get_logger(__name__)


def _load_patch_array(patch_path: Path) -> np.ndarray:
    """Load a patch array from .npy or .npz file."""
    if patch_path.suffix == ".npz":
        with np.load(patch_path) as data:
            if "patch" in data:
                return data["patch"]
            return data[list(data.files)[0]]
    return np.load(patch_path)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
# Model band positions (0-indexed)
# [B01, B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
              "B08", "B8A", "B11", "B12"]

# Key band indices
B04_IDX = 3   # Red
B06_IDX = 5   # Vegetation Red Edge
B08_IDX = 7   # NIR
B8A_IDX = 8   # Narrow NIR
B11_IDX = 9   # SWIR 1

# Per-band z-score stats that Stage 2 uses — must stay in sync with
# pipeline/02_preprocess.py BAND_MEANS / BAND_STDS.
# These are applied in reverse here to recover raw reflectance values.
BAND_MEANS = np.array([0.057, 0.054, 0.046, 0.036, 0.033,
                       0.041, 0.049, 0.043, 0.050, 0.031, 0.019],
                      dtype=np.float32)
BAND_STDS  = np.array([0.010, 0.010, 0.013, 0.010, 0.012,
                       0.020, 0.030, 0.020, 0.030, 0.020, 0.013],
                      dtype=np.float32)
# Band positions that were zero-padded during preprocessing (not downloaded)
# and therefore carry no real spectral information.
ZERO_PADDED_BANDS = [0, 5, 6]  # B01, B06, B07

# Wavelength centers (nm) for FDI calculation
WL_B06 = 740
WL_B08 = 832
WL_B11 = 1610
WL_RED = 665  # B04


# ─────────────────────────────────────────────
# NODATA HELPER
# ─────────────────────────────────────────────
def _compute_nodata_fraction(
    geom: Any,
    nodata_mask: np.ndarray,
    transform: Affine,
) -> float:
    """Return the fraction of pixels inside *geom* that are nodata.

    Returns 1.0 when the geometry falls entirely outside the raster or
    masking fails.
    """
    from rasterio.features import geometry_mask

    h, w = nodata_mask.shape
    try:
        inside = geometry_mask(
            [mapping(geom)],
            out_shape=(h, w),
            transform=transform,
            invert=True,
        )
    except Exception:
        return 1.0

    if not inside.any():
        return 1.0

    return float(nodata_mask[inside].sum()) / float(inside.sum())


# ─────────────────────────────────────────────
# SPECTRAL INDICES
# ─────────────────────────────────────────────
def compute_spectral_indices(
    mean_spectrum: np.ndarray,
) -> Dict[str, float]:
    """Compute polymer-discriminating spectral indices.

    Args:
        mean_spectrum: 11-element array of mean reflectance values
            for a debris cluster, in model band order.

    Returns:
        Dict with keys ``pi``, ``sr``, ``nsi``, ``fdi``.
    """
    eps = 1e-8

    b04 = mean_spectrum[B04_IDX]
    b06 = mean_spectrum[B06_IDX]
    b08 = mean_spectrum[B08_IDX]
    b8a = mean_spectrum[B8A_IDX]
    b11 = mean_spectrum[B11_IDX]

    # Plastic Index: (B08 - B04) / (B08 + B04)
    pi = (b08 - b04) / (b08 + b04 + eps)

    # SWIR Ratio: B11 / B08
    sr = b11 / (b08 + eps)

    # NIR-SWIR Index: (B8A - B11) / (B8A + B11)
    nsi = (b8a - b11) / (b8a + b11 + eps)

    # Floating Debris Index:
    # FDI = B08 - (B06 + (B11-B06) * ((832-665)/(1610-665)))
    interpolation = (WL_B08 - WL_RED) / (WL_B11 - WL_RED + eps)
    fdi = b08 - (b06 + (b11 - b06) * interpolation)

    return {
        "pi": float(pi),
        "sr": float(sr),
        "nsi": float(nsi),
        "fdi": float(fdi),
    }


def classify_polymer(indices: Dict[str, float]) -> Tuple[str, bool]:
    """Classify polymer type using a rule-based spectral classifier.

    Args:
        indices: Dict from :func:`compute_spectral_indices`.

    Returns:
        Tuple of ``(polymer_type, is_false_positive)``.
    """
    pi = indices["pi"]
    sr = indices["sr"]
    nsi = indices["nsi"]
    fdi = indices["fdi"]

    # Organic matter (seaweed, foam) — flag as false positive
    # Relaxed to 0.35 to reduce over-filtering; NDI > 0.35 is strong vegetation signal
    if nsi > 0.35:
        return "Organic Matter", True

    # Polyethylene (PE) / Polypropylene (PP) — most common nurdles
    if pi > 0.1 and sr < 0.3:
        return "PE/PP (Polyethylene/Polypropylene)", False

    # PET / Nylon — fishing nets/lines
    if pi > 0.1 and sr > 0.5:
        return "PET/Nylon", False

    # Mixed polymer / degraded plastic
    if pi > 0.05 and nsi < 0:
        return "Mixed/Degraded Polymer", False

    # Floating debris confirmed by FDI but no clear polymer match
    if fdi > 0.05:
        return "Unidentified Plastic", False

    # Low confidence / ambiguous
    return "Unclassified Debris", False


# ─────────────────────────────────────────────
# SPECTRAL EXTRACTION
# ─────────────────────────────────────────────
def extract_cluster_spectra(
    cluster_geom: Any,
    scene_data: np.ndarray,
    transform: Affine,
) -> Optional[np.ndarray]:
    """Extract mean spectral signature for a debris cluster.

    Args:
        cluster_geom: Shapely geometry of the cluster.
        scene_data: ``(11, H, W)`` raw (un-normalized) scene data.
        transform: Affine geotransform.

    Returns:
        11-element mean spectrum array, or ``None`` if the cluster
        falls outside the scene extent.
    """
    from rasterio.features import geometry_mask

    h, w = scene_data.shape[1], scene_data.shape[2]

    try:
        mask = geometry_mask(
            [mapping(cluster_geom)],
            out_shape=(h, w),
            transform=transform,
            invert=True,  # True inside the geometry
        )
    except Exception:
        return None

    if not mask.any():
        return None

    # Extract mean spectrum across the masked region
    spectra = scene_data[:, mask].mean(axis=1)
    return spectra


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────
def run(
    scene_id: str,
    detections_path: Union[str, Path],
    processed_dir: Union[str, Path],
    output_dir: Union[str, Path] = "data/detections",
    config: Optional[Dict] = None,
) -> Tuple[Path, Dict[str, int]]:
    """Run polymer classification on all detected debris clusters.

    Args:
        scene_id: Scene identifier.
        detections_path: Path to ``detections.geojson`` from Stage 3.
        processed_dir: Directory with preprocessed scene data (patches).
        output_dir: Root output directory for classified detections.
        config: Optional config dict.

    Returns:
        Tuple of (path to ``detections_classified.geojson``,
        dict of polymer type counts).

    Raises:
        FileNotFoundError: If *detections_path* does not exist.
    """
    detections_path = Path(detections_path)
    processed_dir = Path(processed_dir)
    out_dir = Path(output_dir) / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    out_path = out_dir / "detections_classified.geojson"
    if stage_output_exists(out_dir, ["detections_classified.geojson"]):
        gdf = gpd.read_file(out_path)
        counts = gdf["polymer_type"].value_counts().to_dict() if len(gdf) > 0 else {}
        return out_path, counts

    # Load detections
    gdf = gpd.read_file(detections_path)

    # Load the raw (un-normalized) scene for spectral extraction
    # First try to reconstruct from patches
    scene_meta_path = processed_dir / "scene_meta.json"
    scene_data = None
    scene_transform = None
    nodata_mask_2d: Optional[np.ndarray] = None

    if scene_meta_path.exists():
        with open(scene_meta_path) as fh:
            scene_meta = json.load(fh)
        shape_info = scene_meta["original_shape"]
        scene_crs = scene_meta.get("crs")

        # Load patch index and reconstruct
        index_path = processed_dir / "patch_index.json"
        if index_path.exists():
            with open(index_path) as fh:
                patch_index = json.load(fh)

            h, w = shape_info[1], shape_info[2]
            scene_data = np.zeros((shape_info[0], h, w), dtype=np.float32)
            count = np.zeros((h, w), dtype=np.float32)

            for patch_id, info in sorted(patch_index.items()):
                patch_file = info.get("patch_file", f"{patch_id}.npy")
                patch_path = processed_dir / "patches" / patch_file
                if not patch_path.exists():
                    npz_fallback = processed_dir / "patches" / f"{patch_id}.npz"
                    npy_fallback = processed_dir / "patches" / f"{patch_id}.npy"
                    if npz_fallback.exists():
                        patch_path = npz_fallback
                    elif npy_fallback.exists():
                        patch_path = npy_fallback
                if patch_path.exists():
                    patch = _load_patch_array(patch_path)
                    rs = info["row_start"]
                    cs = info["col_start"]
                    ah = info["actual_h"]
                    aw = info["actual_w"]
                    scene_data[:, rs:rs+ah, cs:cs+aw] += patch[:, :ah, :aw]
                    count[rs:rs+ah, cs:cs+aw] += 1.0

            count = np.maximum(count, 1.0)
            scene_data /= count[np.newaxis, :, :]

            # --- Denormalize: patches store z-score values; spectral
            # thresholds expect raw surface reflectance (0–1 scale).
            # raw = normalized * std + mean  (per band)
            # Nodata pixels were zeroed in Stage 2 and must remain 0.
            nodata_mask_path = processed_dir / "nodata_mask.npy"
            if nodata_mask_path.exists():
                nodata_mask_2d = np.load(nodata_mask_path)

            valid = ~nodata_mask_2d if nodata_mask_2d is not None else np.ones(
                (h, w), dtype=bool
            )
            for b_idx in range(scene_data.shape[0]):
                scene_data[b_idx, valid] = (
                    scene_data[b_idx, valid] * BAND_STDS[b_idx] + BAND_MEANS[b_idx]
                )
                scene_data[b_idx, ~valid] = 0.0

            # Zero-padded bands have no real signal — reset to 0 so that
            # the spectral formulae that use them (e.g. FDI via B06) do
            # not produce artefacts from accidental non-zero averaged values.
            for b_idx in ZERO_PADDED_BANDS:
                scene_data[b_idx] = 0.0

            tf_list = scene_meta.get("transform")
            if tf_list:
                scene_transform = Affine(*tf_list[:6])
            else:
                scene_transform = Affine(10, 0, 0, 0, -10, 0)
    else:
        scene_crs = None

    if len(gdf) == 0:
        logger.info("No detections to classify — saving empty file")
        gdf["polymer_type"] = []
        gdf["pi_value"] = []
        gdf["sr_value"] = []
        gdf["nsi_value"] = []
        gdf["fdi_value"] = []
        gdf["is_false_positive"] = []
        gdf.to_file(out_path, driver="GeoJSON")
        return out_path, {}

    gdf_for_spectra = gdf
    if scene_crs and gdf.crs and str(gdf.crs) != str(scene_crs):
        gdf_for_spectra = gdf.to_crs(scene_crs)

    # Classify each cluster
    polymer_types = []
    pi_values = []
    sr_values = []
    nsi_values = []
    fdi_values = []
    is_false_positives = []

    n_nodata = 0
    total_clusters = len(gdf)
    for idx, row in gdf.iterrows():
        spectral_geom = gdf_for_spectra.geometry.iloc[idx]

        # -----------------------------------------------------------
        # Guard: detections whose footprint is mostly nodata are false
        # positives produced by the model on unobserved pixels.
        # -----------------------------------------------------------
        if (
            nodata_mask_2d is not None
            and scene_transform is not None
            and _compute_nodata_fraction(spectral_geom, nodata_mask_2d, scene_transform) > 0.7
        ):
            polymer_types.append("No Data Region")
            pi_values.append(0.0)
            sr_values.append(0.0)
            nsi_values.append(0.0)
            fdi_values.append(0.0)
            is_false_positives.append(True)
            n_nodata += 1
            continue

        if scene_data is not None and scene_transform is not None:
            spectrum = extract_cluster_spectra(
                spectral_geom, scene_data, scene_transform,
            )
        else:
            spectrum = None

        if spectrum is not None:
            indices = compute_spectral_indices(spectrum)
            polymer, is_fp = classify_polymer(indices)
        else:
            # Cannot extract spectrum — use defaults
            indices = {"pi": 0.0, "sr": 0.0, "nsi": 0.0, "fdi": 0.0}
            polymer = "Unknown (insufficient spectral data)"
            is_fp = False

        polymer_types.append(polymer)
        pi_values.append(indices["pi"])
        sr_values.append(indices["sr"])
        nsi_values.append(indices["nsi"])
        fdi_values.append(indices["fdi"])
        is_false_positives.append(is_fp)

        if (idx + 1) % 200 == 0 or idx == total_clusters - 1:
            logger.info(
                "  Processed %d/%d clusters (%d nodata)",
                idx + 1,
                total_clusters,
                n_nodata,
            )

    gdf["polymer_type"] = polymer_types
    gdf["pi_value"] = pi_values
    gdf["sr_value"] = sr_values
    gdf["nsi_value"] = nsi_values
    gdf["fdi_value"] = fdi_values
    gdf["is_false_positive"] = is_false_positives

    gdf.to_file(out_path, driver="GeoJSON")

    # Summary counts
    counts = gdf["polymer_type"].value_counts().to_dict()
    n_fp = sum(is_false_positives)

    logger.info(
        "[bold green]Stage 4 complete[/] — %d clusters classified, %d false positives (%d nodata regions)",
        len(gdf), n_fp, n_nodata,
    )
    if n_nodata == len(gdf) and len(gdf) > 0:
        logger.warning(
            "All Stage 3 detections were flagged as 'No Data Region'; downstream source regions may be empty"
        )
    for ptype, cnt in counts.items():
        logger.info("  %s: %d", ptype, cnt)

    return out_path, counts


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    """CLI entrypoint for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Stage 4: Classify debris clusters by polymer type",
    )
    parser.add_argument("--scene_id", type=str, required=True)
    parser.add_argument("--detections", type=str, required=True,
                        help="Path to detections.geojson from Stage 3")
    parser.add_argument("--processed_dir", type=str, required=True,
                        help="Path to processed scene dir")
    parser.add_argument("--output_dir", type=str, default="data/detections")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    out_path, counts = run(
        scene_id=args.scene_id,
        detections_path=args.detections,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        config=config,
    )
    print(f"\nClassified detections saved to {out_path}")
    print(f"Polymer distribution: {counts}")


if __name__ == "__main__":
    main()

"""
Plastic-Ledger — Stage 3: Marine Debris Detection
====================================================
Runs the trained MARIDA U-Net model on preprocessed patches and produces
georeferenced debris detections with TTA and clustering.

Usage (standalone):
    python -m pipeline.03_detect \\
        --scene_id SCENE_ID \\
        --patches_dir data/processed/SCENE_ID/patches \\
        --model_path models/runs/marida_v1/best_model.pth \\
        --output_dir data/detections

Dependencies: torch, segmentation_models_pytorch, rasterio, scipy, geopandas, numpy
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import label as ndimage_label
import geopandas as gpd
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
import segmentation_models_pytorch as smp

from pipeline.utils.logging_utils import get_logger
from pipeline.utils.geo_utils import save_geotiff, array_to_polygons
from pipeline.utils.cache_utils import load_config, stage_output_exists

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
NUM_CLASSES = 15
NUM_BANDS = 11
DEBRIS_CLASS_INDEX = 0
DEFAULT_THRESHOLD = 0.3
MIN_CLUSTER_AREA_M2 = 100  # 4 pixels at 10m resolution

CLASS_MAP = {
    0: "Marine Debris", 1: "Dense Sargassum", 2: "Sparse Sargassum",
    3: "Natural Organic Material", 4: "Ship", 5: "Clouds",
    6: "Marine Water", 7: "Sediment-Laden Water", 8: "Foam",
    9: "Turbid Water", 10: "Shallow Water", 11: "Waves",
    12: "Cloud Shadows", 13: "Wakes", 14: "Mixed Water",
}


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def load_model(
    checkpoint_path: Union[str, Path],
    device: torch.device,
) -> torch.nn.Module:
    """Load the trained U-Net model from a checkpoint.

    Args:
        checkpoint_path: Path to the ``.pth`` checkpoint file.
        device: Torch device to load the model onto.

    Returns:
        Model in eval mode.

    Raises:
        FileNotFoundError: If *checkpoint_path* does not exist.
        KeyError: If the checkpoint is missing required keys.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    encoder = ckpt.get("encoder", "resnet34")
    num_bands = ckpt.get("num_bands", NUM_BANDS)
    num_classes = ckpt.get("num_classes", NUM_CLASSES)

    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=num_bands,
        classes=num_classes,
        activation=None,
    ).to(device).float()

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    logger.info(
        "Loaded model: encoder=%s, epoch=%s, bands=%d, classes=%d",
        encoder, ckpt.get("epoch", "?"), num_bands, num_classes,
    )
    return model


# ─────────────────────────────────────────────
# TTA INFERENCE
# ─────────────────────────────────────────────
def _apply_augmentation(patch: np.ndarray, aug_type: str) -> np.ndarray:
    """Apply a single augmentation transform to a patch.

    Args:
        patch: ``(C, H, W)`` array.
        aug_type: One of ``original``, ``hflip``, ``vflip``,
            ``rot90``, ``rot180``, ``rot270``.

    Returns:
        Augmented ``(C, H, W)`` array.
    """
    if aug_type == "original":
        return patch
    elif aug_type == "hflip":
        return patch[:, :, ::-1].copy()
    elif aug_type == "vflip":
        return patch[:, ::-1, :].copy()
    elif aug_type == "rot90":
        return np.rot90(patch, k=1, axes=(1, 2)).copy()
    elif aug_type == "rot180":
        return np.rot90(patch, k=2, axes=(1, 2)).copy()
    elif aug_type == "rot270":
        return np.rot90(patch, k=3, axes=(1, 2)).copy()
    return patch


def _reverse_augmentation(probs: np.ndarray, aug_type: str) -> np.ndarray:
    """Reverse an augmentation on a probability map.

    Args:
        probs: ``(C, H, W)`` probability array.
        aug_type: The augmentation that was applied.

    Returns:
        De-augmented ``(C, H, W)`` array.
    """
    if aug_type == "original":
        return probs
    elif aug_type == "hflip":
        return probs[:, :, ::-1].copy()
    elif aug_type == "vflip":
        return probs[:, ::-1, :].copy()
    elif aug_type == "rot90":
        return np.rot90(probs, k=-1, axes=(1, 2)).copy()
    elif aug_type == "rot180":
        return np.rot90(probs, k=-2, axes=(1, 2)).copy()
    elif aug_type == "rot270":
        return np.rot90(probs, k=-3, axes=(1, 2)).copy()
    return probs


@torch.no_grad()
def run_tta_inference(
    model: torch.nn.Module,
    patch: np.ndarray,
    device: torch.device,
    use_tta: bool = True,
) -> np.ndarray:
    """Run inference with Test-Time Augmentation (6 variants).

    Args:
        model: Loaded U-Net model in eval mode.
        patch: ``(C, H, W)`` normalized float32 array.
        device: Torch device.
        use_tta: If ``False``, skip TTA and run single forward pass.

    Returns:
        ``(num_classes, H, W)`` averaged probability map.
    """
    aug_types = (
        ["original", "hflip", "vflip", "rot90", "rot180", "rot270"]
        if use_tta
        else ["original"]
    )

    accumulated = None

    for aug in aug_types:
        augmented = _apply_augmentation(patch, aug)
        tensor = torch.from_numpy(augmented).unsqueeze(0).float().to(device)

        logits = model(tensor)
        logits = torch.clamp(logits, -30.0, 30.0)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()  # (C, H, W)

        # Reverse the augmentation on the output
        probs = _reverse_augmentation(probs, aug)

        if accumulated is None:
            accumulated = probs
        else:
            accumulated += probs

    return accumulated / len(aug_types)


# ─────────────────────────────────────────────
# PATCH STITCHING
# ─────────────────────────────────────────────
def stitch_patches(
    predictions: List[np.ndarray],
    patch_index: Dict[str, Dict],
    scene_shape: Tuple[int, int],
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Stitch patch predictions back into a full-scene probability map.

    Overlap regions are averaged.

    Args:
        predictions: List of ``(num_classes, H, W)`` probability arrays.
        patch_index: Patch index dict mapping patch_id → info.
        scene_shape: ``(height, width)`` of the original scene.
        num_classes: Number of output classes.

    Returns:
        ``(num_classes, H, W)`` full-scene probability map.
    """
    h, w = scene_shape
    prob_sum = np.zeros((num_classes, h, w), dtype=np.float16)
    count = np.zeros((h, w), dtype=np.float16)

    patch_ids = sorted(patch_index.keys())
    for patch_id, pred in zip(patch_ids, predictions):
        info = patch_index[patch_id]
        rs = info["row_start"]
        cs = info["col_start"]
        ah = info["actual_h"]
        aw = info["actual_w"]

        prob_sum[:, rs:rs + ah, cs:cs + aw] += pred[:, :ah, :aw]
        count[rs:rs + ah, cs:cs + aw] += 1.0

    # Avoid division by zero
    count = np.maximum(count, 1.0)
    prob_map = prob_sum / count[np.newaxis, :, :]

    return prob_map.astype(np.float32)


# ─────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────
def extract_clusters(
    debris_mask: np.ndarray,
    debris_prob: np.ndarray,
    transform: Affine,
    crs: Any,
    min_area_m2: float = MIN_CLUSTER_AREA_M2,
    detection_date: str = "",
) -> gpd.GeoDataFrame:
    """Cluster connected debris pixels into detection objects.

    Args:
        debris_mask: ``(H, W)`` boolean mask of debris pixels.
        debris_prob: ``(H, W)`` debris probability map.
        transform: Affine geotransform.
        crs: Coordinate reference system.
        min_area_m2: Minimum cluster area in m².
        detection_date: ISO date string for the detection.

    Returns:
        :class:`GeoDataFrame` with columns: geometry, area_m2,
        mean_confidence, centroid_lon, centroid_lat, detection_date, cluster_id.
    """
    if not debris_mask.any():
        logger.info("No debris pixels detected — returning empty GeoDataFrame")
        return gpd.GeoDataFrame(
            columns=["geometry", "area_m2", "mean_confidence",
                     "centroid_lon", "centroid_lat", "detection_date",
                     "cluster_id"],
            crs=crs if crs else "EPSG:4326",
        )

    labeled, num_features = ndimage_label(debris_mask.astype(np.int32))
    logger.info("Found %d raw connected components", num_features)

    # Pixel resolution in meters (approximate from transform)
    pixel_area_m2 = abs(transform.a * transform.e)
    min_pixels = max(1, int(min_area_m2 / pixel_area_m2))

    clusters = []
    for cluster_id in range(1, num_features + 1):
        cluster_mask = (labeled == cluster_id)
        n_pixels = cluster_mask.sum()

        if n_pixels < min_pixels:
            continue

        area_m2 = float(n_pixels * pixel_area_m2)
        mean_conf = float(debris_prob[cluster_mask].mean())

        # Convert cluster mask to polygon
        gdf = array_to_polygons(
            cluster_mask.astype(np.uint8), transform, crs,
        )
        if gdf.empty:
            continue

        # Merge all polygons for this cluster
        merged_geom = gdf.unary_union
        centroid = merged_geom.centroid

        # Re-project centroid to lon/lat if needed
        if crs and str(crs) != "EPSG:4326":
            import pyproj
            from shapely.ops import transform as shp_transform
            project = pyproj.Transformer.from_crs(
                crs, "EPSG:4326", always_xy=True,
            ).transform
            centroid_lonlat = shp_transform(project, centroid)
        else:
            centroid_lonlat = centroid

        clusters.append({
            "geometry": merged_geom,
            "area_m2": area_m2,
            "mean_confidence": mean_conf,
            "centroid_lon": centroid_lonlat.x,
            "centroid_lat": centroid_lonlat.y,
            "detection_date": detection_date,
            "cluster_id": len(clusters),
        })

    if not clusters:
        logger.info("No clusters above minimum area (%d m²)", min_area_m2)
        return gpd.GeoDataFrame(
            columns=["geometry", "area_m2", "mean_confidence",
                     "centroid_lon", "centroid_lat", "detection_date",
                     "cluster_id"],
            crs=crs if crs else "EPSG:4326",
        )

    gdf = gpd.GeoDataFrame(clusters, crs=crs if crs else "EPSG:4326")
    logger.info(
        "Extracted %d debris clusters (filtered from %d components, min %d m²)",
        len(gdf), num_features, min_area_m2,
    )
    return gdf


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────
def run(
    scene_id: str,
    patches_dir: Union[str, Path],
    model_path: Union[str, Path],
    output_dir: Union[str, Path] = "data/detections",
    config: Optional[Dict] = None,
) -> Path:
    """Run the full detection stage for one scene.

    Args:
        scene_id: Identifier of the scene being processed.
        patches_dir: Directory with ``.npy`` patch files.
        model_path: Path to the trained model checkpoint.
        output_dir: Root output directory for detections.
        config: Optional config dict.

    Returns:
        Path to the output ``detections.geojson``.

    Raises:
        FileNotFoundError: If patches or model cannot be found.
    """
    patches_dir = Path(patches_dir)
    out_dir = Path(output_dir) / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Settings
    threshold = DEFAULT_THRESHOLD
    use_tta = True
    min_area = MIN_CLUSTER_AREA_M2
    if config:
        model_cfg = config.get("model", {})
        threshold = model_cfg.get("debris_threshold", DEFAULT_THRESHOLD)
        use_tta = model_cfg.get("tta", True)
        det_cfg = config.get("detection", {})
        min_area = det_cfg.get("min_cluster_area_m2", MIN_CLUSTER_AREA_M2)

    # Check cache
    if stage_output_exists(out_dir, ["detections.geojson"]):
        return out_dir / "detections.geojson"

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    # Load patch index
    processed_dir = patches_dir.parent
    index_path = processed_dir / "patch_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Patch index not found: {index_path}")
    with open(index_path) as fh:
        patch_index = json.load(fh)

    # Load scene metadata for shape/CRS
    meta_path = processed_dir / "scene_meta.json"
    if meta_path.exists():
        with open(meta_path) as fh:
            scene_meta = json.load(fh)
        scene_shape = (scene_meta["original_shape"][1],
                       scene_meta["original_shape"][2])
        scene_crs = scene_meta.get("crs")
        scene_transform_list = scene_meta.get("transform")
        if scene_transform_list:
            scene_transform = Affine(*scene_transform_list[:6])
        else:
            scene_transform = Affine(10, 0, 0, 0, -10, 0)  # fallback 10m
    else:
        # Infer from patches
        max_row_start = max(p["row_start"] + p["actual_h"]
                           for p in patch_index.values())
        max_col_start = max(p["col_start"] + p["actual_w"]
                           for p in patch_index.values())
        scene_shape = (max_row_start, max_col_start)
        scene_crs = None
        scene_transform = Affine(10, 0, 0, 0, -10, 0)

    # Run inference on each patch
    logger.info("Running inference on %d patches (TTA=%s)", len(patch_index), use_tta)

    predictions = []
    patch_ids = sorted(patch_index.keys())

    for i, patch_id in enumerate(patch_ids):
        patch_path = patches_dir / f"{patch_id}.npy"
        if not patch_path.exists():
            logger.warning("Patch file missing: %s", patch_path)
            # Create zero prediction
            predictions.append(np.zeros((NUM_CLASSES, 256, 256), dtype=np.float32))
            continue

        patch = np.load(patch_path)
        prob_map = run_tta_inference(model, patch, device, use_tta=use_tta)
        predictions.append(prob_map)

        if (i + 1) % 50 == 0 or i == len(patch_ids) - 1:
            logger.info("  Processed %d/%d patches", i + 1, len(patch_ids))

    # Stitch predictions
    logger.info("Stitching %d patch predictions into full scene", len(predictions))
    full_probs = stitch_patches(predictions, patch_index, scene_shape)

    # Generate masks
    debris_prob = full_probs[DEBRIS_CLASS_INDEX]
    class_mask = full_probs.argmax(axis=0).astype(np.uint8)
    debris_mask = (debris_prob > threshold) & (class_mask == DEBRIS_CLASS_INDEX)

    # Save outputs
    profile_base = {
        "driver": "GTiff",
        "height": scene_shape[0],
        "width": scene_shape[1],
        "transform": scene_transform,
        "crs": scene_crs,
    }

    # Binary debris mask
    save_geotiff(
        out_dir / "debris_mask.tif",
        debris_mask.astype(np.uint8),
        profile_base,
        dtype="uint8",
    )

    # Debris probability map
    save_geotiff(
        out_dir / "debris_prob.tif",
        debris_prob,
        profile_base,
        dtype="float32",
    )

    # Full class mask
    save_geotiff(
        out_dir / "class_mask.tif",
        class_mask,
        profile_base,
        dtype="uint8",
    )

    # Get detection date from scene metadata if available
    detection_date = ""
    ingest_meta = processed_dir.parent.parent / "raw" / scene_id / "metadata.json"
    if ingest_meta.exists():
        with open(ingest_meta) as fh:
            img_meta = json.load(fh)
        detection_date = img_meta.get("datetime", "")

    # Extract clusters as GeoJSON
    gdf = extract_clusters(
        debris_mask, debris_prob, scene_transform, scene_crs,
        min_area_m2=min_area, detection_date=detection_date,
    )

    geojson_path = out_dir / "detections.geojson"
    geojson_gdf = gdf
    if not gdf.empty and gdf.crs and str(gdf.crs) != "EPSG:4326":
        geojson_gdf = gdf.to_crs("EPSG:4326")
    geojson_gdf.to_file(geojson_path, driver="GeoJSON")

    logger.info(
        "[bold green]Stage 3 complete[/] — %d clusters, debris pixels=%d (%.3f%%)",
        len(gdf),
        debris_mask.sum(),
        100 * debris_mask.sum() / debris_mask.size,
    )
    return geojson_path


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    """CLI entrypoint for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Stage 3: Run marine debris detection on preprocessed patches",
    )
    parser.add_argument("--scene_id", type=str, required=True)
    parser.add_argument("--patches_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str,
                        default="models/runs/marida_v1/best_model.pth")
    parser.add_argument("--output_dir", type=str, default="data/detections")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    geojson_path = run(
        scene_id=args.scene_id,
        patches_dir=args.patches_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        config=config,
    )
    print(f"\nDetections saved to {geojson_path}")


if __name__ == "__main__":
    main()

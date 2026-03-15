"""
Plastic-Ledger — Master Pipeline Runner
==========================================
Orchestrates all 7 stages end-to-end with progress logging,
stage caching, and graceful error handling.

Usage:
    python pipeline/run_pipeline.py \\
        --bbox "80.0,8.0,82.0,10.0" \\
        --start_date "2024-01-01" \\
        --end_date "2024-01-31" \\
        --output_dir "data/runs/run_001" \\
        --model_path "models/runs/marida_v1/best_model.pth" \\
        --cloud_cover 20 \\
        --backtrack_days 30 \\
        --skip_stages ""
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pipeline.utils.logging_utils import get_logger
from pipeline.utils.cache_utils import load_config

logger = get_logger(__name__)


def _parse_skip_stages(skip_str: str) -> Set[int]:
    """Parse comma-separated stage numbers to skip.

    Args:
        skip_str: e.g. ``"1,5"`` or ``""``.

    Returns:
        Set of stage numbers to skip.
    """
    if not skip_str or not skip_str.strip():
        return set()
    return {int(s.strip()) for s in skip_str.split(",") if s.strip()}


def run_pipeline(
    bbox: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    output_dir: Union[str, Path],
    model_path: Union[str, Path],
    cloud_cover: int = 20,
    backtrack_days: int = 30,
    skip_stages: Set[int] = None,
    config_path: str = "config/config.yaml",
) -> Dict[str, Any]:
    """Run the complete Plastic-Ledger pipeline.

    Args:
        bbox: ``(lon_min, lat_min, lon_max, lat_max)``.
        start_date: Start date ISO string.
        end_date: End date ISO string.
        output_dir: Root output directory for this run.
        model_path: Path to the trained model checkpoint.
        cloud_cover: Maximum cloud cover percentage.
        backtrack_days: Days to back-track particles.
        skip_stages: Set of stage numbers (1–7) to skip.
        config_path: Path to YAML config file.

    Returns:
        Dict with run summary including output paths and metrics.

    Raises:
        RuntimeError: If critical stages fail.
    """
    if skip_stages is None:
        skip_stages = set()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path)

    # Load config
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        logger.warning("Config file not found: %s — using defaults", config_path)
        config = {}

    # Override backtracking days in config
    if "backtracking" not in config:
        config["backtracking"] = {}
    config["backtracking"]["days"] = backtrack_days

    # Create subdirectories
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    detections_dir = output_dir / "detections"
    attribution_dir = output_dir / "attribution"
    reports_dir = output_dir / "reports"

    for d in [raw_dir, processed_dir, detections_dir, attribution_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    run_summary = {
        "bbox": list(bbox),
        "start_date": start_date,
        "end_date": end_date,
        "model_path": str(model_path),
        "stages_completed": [],
        "stages_skipped": list(skip_stages),
        "stages_failed": [],
        "outputs": {},
    }

    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        console.print(Panel(
            "[bold cyan]🌊 Plastic-Ledger Pipeline[/]\n"
            f"  BBox: {bbox}\n"
            f"  Dates: {start_date} → {end_date}\n"
            f"  Cloud Cover: ≤{cloud_cover}%\n"
            f"  Back-track: {backtrack_days} days\n"
            f"  Skip Stages: {skip_stages or 'none'}",
            style="cyan",
        ))
    except ImportError:
        print(f"\n{'='*60}")
        print(f"  Plastic-Ledger Pipeline")
        print(f"  BBox: {bbox}, Dates: {start_date}-{end_date}")
        print(f"{'='*60}")

    scene_dirs = []
    scene_metas = []
    t_start = time.time()

    # ═══════════════════════════════════════════
    # STAGE 1 — Satellite Data Ingestion
    # ═══════════════════════════════════════════
    if 1 not in skip_stages:
        logger.info("[bold blue]━━━ Stage 1: Satellite Data Ingestion ━━━[/]")
        try:
            from pipeline.01_ingest import run as ingest_run
            scene_dirs, scene_metas = ingest_run(
                bbox=bbox,
                date_start=start_date,
                date_end=end_date,
                cloud_cover_max=cloud_cover,
                output_dir=str(raw_dir),
                config=config,
            )
            run_summary["stages_completed"].append(1)
            run_summary["outputs"]["raw_scenes"] = [str(d) for d in scene_dirs]
        except Exception as exc:
            logger.error("Stage 1 failed: %s", exc)
            run_summary["stages_failed"].append({"stage": 1, "error": str(exc)})
    else:
        logger.info("[dim]Stage 1 skipped[/]")
        # Try to find existing scenes
        if raw_dir.exists():
            scene_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]

    # If no scenes, try to continue with existing processed data
    if not scene_dirs:
        if processed_dir.exists():
            scene_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
            if scene_dirs:
                logger.info(
                    "No raw scenes but found %d processed scenes", len(scene_dirs),
                )
        if not scene_dirs:
            logger.warning("No scenes available — pipeline cannot continue")
            _save_summary(run_summary, output_dir, t_start)
            return run_summary

    # Process each scene through stages 2-7
    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        logger.info("\n[bold magenta]Processing scene: %s[/]", scene_id)

        # Scene-level paths
        scene_processed = processed_dir / scene_id
        patches_dir = scene_processed / "patches"
        scene_detections = detections_dir / scene_id
        scene_attribution = attribution_dir / scene_id

        detection_date = None
        meta_path = scene_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as fh:
                meta = json.load(fh)
            detection_date = meta.get("datetime")

        # ═══════════════════════════════════════
        # STAGE 2 — Preprocessing
        # ═══════════════════════════════════════
        if 2 not in skip_stages:
            logger.info("[bold blue]━━━ Stage 2: Preprocessing ━━━[/]")
            try:
                from pipeline.02_preprocess import run as preprocess_run
                patches_dir, patch_index = preprocess_run(
                    scene_dir=str(scene_dir),
                    output_dir=str(processed_dir),
                    config=config,
                )
                run_summary["stages_completed"].append(2)
            except Exception as exc:
                logger.error("Stage 2 failed for %s: %s", scene_id, exc)
                run_summary["stages_failed"].append(
                    {"stage": 2, "scene": scene_id, "error": str(exc)},
                )
                continue
        else:
            logger.info("[dim]Stage 2 skipped[/]")

        # ═══════════════════════════════════════
        # STAGE 3 — Marine Debris Detection
        # ═══════════════════════════════════════
        geojson_path = scene_detections / "detections.geojson"
        if 3 not in skip_stages:
            logger.info("[bold blue]━━━ Stage 3: Marine Debris Detection ━━━[/]")
            try:
                from pipeline.03_detect import run as detect_run
                geojson_path = detect_run(
                    scene_id=scene_id,
                    patches_dir=str(patches_dir),
                    model_path=str(model_path),
                    output_dir=str(detections_dir),
                    config=config,
                )
                run_summary["stages_completed"].append(3)
            except Exception as exc:
                logger.error("Stage 3 failed for %s: %s", scene_id, exc)
                run_summary["stages_failed"].append(
                    {"stage": 3, "scene": scene_id, "error": str(exc)},
                )
                continue
        else:
            logger.info("[dim]Stage 3 skipped[/]")

        # ═══════════════════════════════════════
        # STAGE 4 — Polymer Classification
        # ═══════════════════════════════════════
        classified_path = scene_detections / "detections_classified.geojson"
        if 4 not in skip_stages:
            logger.info("[bold blue]━━━ Stage 4: Polymer Classification ━━━[/]")
            try:
                from pipeline.04_polymer import run as polymer_run
                classified_path, polymer_counts = polymer_run(
                    scene_id=scene_id,
                    detections_path=str(geojson_path),
                    processed_dir=str(scene_processed),
                    output_dir=str(detections_dir),
                    config=config,
                )
                run_summary["stages_completed"].append(4)
                run_summary["outputs"]["polymer_counts"] = polymer_counts
            except Exception as exc:
                logger.error("Stage 4 failed for %s: %s", scene_id, exc)
                run_summary["stages_failed"].append(
                    {"stage": 4, "scene": scene_id, "error": str(exc)},
                )
                classified_path = geojson_path  # fallback to unclassified
        else:
            logger.info("[dim]Stage 4 skipped[/]")
            if not classified_path.exists():
                classified_path = geojson_path

        # ═══════════════════════════════════════
        # STAGE 5 — Hydrodynamic Back-Tracking
        # ═══════════════════════════════════════
        sources = []
        if 5 not in skip_stages:
            logger.info("[bold blue]━━━ Stage 5: Hydrodynamic Back-Tracking ━━━[/]")
            try:
                from pipeline.05_backtrack import run as backtrack_run
                sources = backtrack_run(
                    scene_id=scene_id,
                    detections_path=str(classified_path),
                    output_dir=str(attribution_dir),
                    config=config,
                    detection_date=detection_date,
                    bbox=bbox,
                )
                run_summary["stages_completed"].append(5)
            except Exception as exc:
                logger.error("Stage 5 failed for %s: %s", scene_id, exc)
                run_summary["stages_failed"].append(
                    {"stage": 5, "scene": scene_id, "error": str(exc)},
                )
        else:
            logger.info("[dim]Stage 5 skipped[/]")
            bt_summary = scene_attribution / "backtrack_summary.json"
            if bt_summary.exists():
                with open(bt_summary) as fh:
                    sources = json.load(fh)

        # ═══════════════════════════════════════
        # STAGE 6 — Source Attribution
        # ═══════════════════════════════════════
        attribution_path = scene_attribution / "attribution_report.json"
        if 6 not in skip_stages:
            logger.info("[bold blue]━━━ Stage 6: Source Attribution ━━━[/]")
            try:
                from pipeline.06_attribute import run as attribute_run
                attribution_path = attribute_run(
                    scene_id=scene_id,
                    sources=sources,
                    detections_path=str(classified_path),
                    output_dir=str(attribution_dir),
                    config=config,
                    detection_date=detection_date,
                )
                run_summary["stages_completed"].append(6)
            except Exception as exc:
                logger.error("Stage 6 failed for %s: %s", scene_id, exc)
                run_summary["stages_failed"].append(
                    {"stage": 6, "scene": scene_id, "error": str(exc)},
                )
        else:
            logger.info("[dim]Stage 6 skipped[/]")

        # ═══════════════════════════════════════
        # STAGE 7 — Report Generation
        # ═══════════════════════════════════════
        if 7 not in skip_stages:
            logger.info("[bold blue]━━━ Stage 7: Report Generation ━━━[/]")
            try:
                from pipeline.07_report import run as report_run
                report_outputs = report_run(
                    scene_id=scene_id,
                    detections_path=str(classified_path),
                    attribution_path=str(attribution_path),
                    output_dir=str(reports_dir),
                    config=config,
                )
                run_summary["stages_completed"].append(7)
                run_summary["outputs"]["reports"] = {
                    k: str(v) for k, v in report_outputs.items()
                }
            except Exception as exc:
                logger.error("Stage 7 failed for %s: %s", scene_id, exc)
                run_summary["stages_failed"].append(
                    {"stage": 7, "scene": scene_id, "error": str(exc)},
                )
        else:
            logger.info("[dim]Stage 7 skipped[/]")

    _save_summary(run_summary, output_dir, t_start)
    return run_summary


def _save_summary(
    summary: Dict[str, Any],
    output_dir: Path,
    t_start: float,
):
    """Save run summary and print final output.

    Args:
        summary: Run summary dict.
        output_dir: Output directory.
        t_start: Pipeline start time.
    """
    elapsed = time.time() - t_start
    summary["elapsed_seconds"] = round(elapsed, 1)

    summary_path = output_dir / "run_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    # Final output
    n_completed = len(set(summary["stages_completed"]))
    n_failed = len(summary["stages_failed"])

    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        console.print(Panel(
            f"[bold green]✅ Pipeline Complete[/]\n"
            f"  Stages completed: {n_completed}\n"
            f"  Stages failed: {n_failed}\n"
            f"  Time elapsed: {elapsed:.1f}s\n"
            f"  Summary: {summary_path}\n"
            f"  Reports: {summary.get('outputs', {}).get('reports', 'N/A')}",
            style="green",
        ))
    except ImportError:
        print(f"\n✅ Pipeline complete: {n_completed} stages, "
              f"{n_failed} failures, {elapsed:.1f}s")
        print(f"   Summary: {summary_path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    """CLI entrypoint for the full pipeline."""
    parser = argparse.ArgumentParser(
        description="Plastic-Ledger: End-to-end marine debris detection pipeline",
    )
    parser.add_argument(
        "--bbox", type=str, required=True,
        help="Bounding box: 'lon_min,lat_min,lon_max,lat_max'",
    )
    parser.add_argument("--start_date", type=str, required=True)
    parser.add_argument("--end_date", type=str, required=True)
    parser.add_argument(
        "--output_dir", type=str, default="data/runs/run_001",
        help="Output directory for this run",
    )
    parser.add_argument(
        "--model_path", type=str,
        default="models/runs/marida_v1/best_model.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument("--cloud_cover", type=int, default=20)
    parser.add_argument("--backtrack_days", type=int, default=30)
    parser.add_argument(
        "--skip_stages", type=str, default="",
        help="Comma-separated stage numbers to skip, e.g. '1,5'",
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    bbox = tuple(float(x) for x in args.bbox.split(","))
    assert len(bbox) == 4, "bbox must have exactly 4 comma-separated values"

    skip = _parse_skip_stages(args.skip_stages)

    summary = run_pipeline(
        bbox=bbox,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        model_path=args.model_path,
        cloud_cover=args.cloud_cover,
        backtrack_days=args.backtrack_days,
        skip_stages=skip,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()

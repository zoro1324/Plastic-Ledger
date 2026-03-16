"""
Plastic-Ledger — Stage 7: Report Generation
===============================================
Generates a complete PDF report, GeoJSON summary, CSV export, and
terminal summary for each processed scene.

Usage (standalone):
    python -m pipeline.07_report \\
        --scene_id SCENE_ID \\
        --detections data/detections/SCENE_ID/detections_classified.geojson \\
        --attribution data/attribution/SCENE_ID/attribution_report.json \\
        --output_dir data/reports

Dependencies: fpdf2, matplotlib, geopandas, pandas, rich
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

from pipeline.utils.logging_utils import get_logger
from pipeline.utils.cache_utils import load_config, stage_output_exists

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# PDF REPORT
# ─────────────────────────────────────────────
def generate_pdf(
    scene_id: str,
    detections_gdf: gpd.GeoDataFrame,
    attribution_data: List[Dict[str, Any]],
    output_dir: Path,
    detection_map_path: Optional[Path] = None,
    trajectory_map_path: Optional[Path] = None,
) -> Path:
    """Generate a PDF report with executive summary, maps, and tables.

    Args:
        scene_id: Scene identifier.
        detections_gdf: GeoDataFrame of classified debris clusters.
        attribution_data: List of attribution report entries.
        output_dir: Output directory.
        detection_map_path: Optional path to a PNG detection map.
        trajectory_map_path: Optional path to a PNG trajectory map.

    Returns:
        Path to the generated PDF file.

    Raises:
        ImportError: If ``fpdf2`` is not installed.
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Page 1: Executive Summary ─────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, "Plastic-Ledger Report", align="C", ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Scene: {scene_id}", align="C", ln=1)
    pdf.ln(10)

    # Summary statistics
    n_clusters = len(detections_gdf)
    total_area = detections_gdf["area_m2"].sum() if n_clusters > 0 and "area_m2" in detections_gdf.columns else 0
    total_area_km2 = total_area / 1e6

    # Dominant polymer
    dominant_polymer = "N/A"
    if n_clusters > 0 and "polymer_type" in detections_gdf.columns:
        polymer_counts = detections_gdf["polymer_type"].value_counts()
        dominant_polymer = polymer_counts.index[0] if len(polymer_counts) > 0 else "N/A"

    # Detection date
    det_date = "Unknown"
    if n_clusters > 0 and "detection_date" in detections_gdf.columns:
        dates = detections_gdf["detection_date"].dropna()
        if len(dates) > 0:
            det_date = str(dates.iloc[0])

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Executive Summary", ln=1)
    pdf.set_font("Helvetica", "", 11)

    summary_lines = [
        f"Detection Date: {det_date}",
        f"Total Debris Clusters: {n_clusters}",
        f"Total Debris Area: {total_area_km2:.4f} sq km  ({total_area:.0f} sq m)",
        f"Dominant Polymer Type: {dominant_polymer}",
    ]
    if attribution_data:
        top_source = attribution_data[0]
        summary_lines.append(
            f"Top Source Attribution: {top_source.get('source_type', 'Unknown')} "
            f"({top_source.get('attribution_score', 0)*100:.0f}% confidence)"
        )

    for line in summary_lines:
        pdf.cell(0, 7, line, ln=1)

    # ── Detection Map ─────────────────────────────
    if detection_map_path and detection_map_path.exists():
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Detection Map", ln=1)
        try:
            pdf.image(str(detection_map_path), w=180)
        except Exception as exc:
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(0, 7, f"(Map image could not be embedded: {exc})", ln=1)

    # ── Page 2: Polymer Distribution ──────────────
    if n_clusters > 0 and "polymer_type" in detections_gdf.columns:
        # Generate pie chart
        pie_path = output_dir / "polymer_distribution.png"
        _generate_polymer_pie(detections_gdf, pie_path)

        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Polymer Distribution", ln=1)
        if pie_path.exists():
            try:
                pdf.image(str(pie_path), w=140)
            except Exception:
                pass

    # ── Source Attribution ─────────────────────────
    if attribution_data:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Source Attribution - Top 3", ln=1)
        pdf.ln(5)

        for i, attr in enumerate(attribution_data[:3]):
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(
                0, 8,
                f"#{i+1}: {attr.get('source_type', 'Unknown')} "
                f"- {attr.get('attribution_score', 0)*100:.0f}% confidence",
                ln=1,
            )
            pdf.set_font("Helvetica", "", 10)
            explanation = attr.get("explanation", "No explanation available.")
            # Word wrap long explanations
            pdf.multi_cell(0, 6, explanation)
            pdf.ln(3)

    # ── Trajectory Map ────────────────────────────
    if trajectory_map_path and trajectory_map_path.exists():
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Back-Track Trajectories", ln=1)
        try:
            pdf.image(str(trajectory_map_path), w=180)
        except Exception:
            pass

    # ── Cluster Table ─────────────────────────────
    if n_clusters > 0:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Debris Cluster Details", ln=1)
        pdf.ln(3)

        # Table header
        pdf.set_font("Helvetica", "B", 8)
        col_widths = [15, 25, 20, 40, 25, 30, 30]
        headers = ["ID", "Area (sq m)", "Confidence", "Polymer", "Lat", "Lon", "Source"]
        for w, h in zip(col_widths, headers):
            pdf.cell(w, 7, h, border=1)
        pdf.ln()

        # Table rows
        pdf.set_font("Helvetica", "", 7)
        for _, row in detections_gdf.iterrows():
            cid = str(row.get("cluster_id", ""))[:4]
            area = f"{row.get('area_m2', 0):.0f}"
            conf = f"{row.get('mean_confidence', 0):.2f}"
            polymer = str(row.get("polymer_type", ""))[:16]
            lat = f"{row.get('centroid_lat', 0):.4f}"
            lon = f"{row.get('centroid_lon', 0):.4f}"

            # Find attribution for this cluster
            src = "N/A"
            if attribution_data:
                for attr in attribution_data:
                    if attr.get("debris_cluster_id") == row.get("cluster_id"):
                        src = attr.get("source_type", "N/A")[:12]
                        break

            vals = [cid, area, conf, polymer, lat, lon, src]
            for w, v in zip(col_widths, vals):
                pdf.cell(w, 6, v, border=1)
            pdf.ln()

    # Save
    pdf_path = output_dir / "final_report.pdf"
    pdf.output(str(pdf_path))
    logger.info("PDF report saved to %s", pdf_path)
    return pdf_path


def _generate_polymer_pie(gdf: gpd.GeoDataFrame, output_path: Path):
    """Generate a polymer distribution pie chart.

    Args:
        gdf: GeoDataFrame with ``polymer_type`` column.
        output_path: Path to save the PNG.
    """
    counts = gdf["polymer_type"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    colors = ["#E63946", "#2A9D8F", "#F4A261", "#3A86FF", "#8338EC",
              "#57CC99", "#E9C46A", "#FF6B6B"]
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=colors[:len(counts)],
        textprops={"color": "white", "fontsize": 9},
    )
    for autotext in autotexts:
        autotext.set_fontsize(8)
        autotext.set_color("white")

    ax.set_title("Polymer Type Distribution",
                 color="white", fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor="#0d1117", bbox_inches="tight")
    plt.close(fig)


def _generate_detection_map(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
):
    """Generate a simple detection map.

    Args:
        gdf: GeoDataFrame of detections.
        output_path: Path to save the PNG.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#1a1a2e")

    plot_gdf = gdf.copy()
    if len(plot_gdf) > 0 and "geometry" in plot_gdf.columns:
        plot_gdf = plot_gdf[plot_gdf.geometry.notna()]
        plot_gdf = plot_gdf[~plot_gdf.geometry.is_empty]

    if len(plot_gdf) > 0:
        bounds = plot_gdf.total_bounds
        finite_bounds = np.all(np.isfinite(bounds))
        height = bounds[3] - bounds[1] if finite_bounds else np.nan
        width = bounds[2] - bounds[0] if finite_bounds else np.nan

        if not finite_bounds or height <= 0 or width <= 0:
            ax.set_aspect("auto")

        # Plot in raw data coordinates to avoid GeoPandas applying geographic
        # aspect logic to mislabeled or projected geometries.
        plot_gdf = plot_gdf.set_crs(None, allow_override=True)

        plot_gdf.plot(
            ax=ax,
            color="#E63946",
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
            aspect="auto",
        )
        ax.set_title(f"Debris Detections ({len(gdf)} clusters)",
                     color="white", fontsize=13, fontweight="bold")
    else:
        ax.set_title("No Debris Detected",
                     color="gray", fontsize=13, fontweight="bold")

    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor="#0d1117", bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────
# GEOJSON & CSV
# ─────────────────────────────────────────────
def generate_geojson_summary(
    detections_gdf: gpd.GeoDataFrame,
    attribution_data: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Generate a combined GeoJSON with detections + attributions.

    Args:
        detections_gdf: GeoDataFrame of classified detections.
        attribution_data: List of attribution report entries.
        output_path: Output file path.

    Returns:
        Path to the saved GeoJSON.
    """
    if len(detections_gdf) == 0:
        # Save empty GeoJSON
        empty_gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
        empty_gdf.to_file(output_path, driver="GeoJSON")
        return output_path

    # Merge attribution data into detections
    gdf = detections_gdf.copy()

    attr_by_cluster = {}
    for attr in attribution_data:
        cid = attr.get("debris_cluster_id")
        if cid is not None:
            attr_by_cluster[cid] = attr

    # Add attribution columns
    for col in ["source_type", "attribution_score", "explanation", "country"]:
        gdf[col] = gdf.apply(
            lambda row: attr_by_cluster.get(
                row.get("cluster_id"), {}
            ).get(col, ""),
            axis=1,
        )

    gdf.to_file(output_path, driver="GeoJSON")
    logger.info("GeoJSON summary saved to %s", output_path)
    return output_path


def generate_csv(
    detections_gdf: gpd.GeoDataFrame,
    attribution_data: List[Dict[str, Any]],
    scene_id: str,
    output_path: Path,
) -> Path:
    """Generate a flat CSV summary of all detections.

    Args:
        detections_gdf: GeoDataFrame of classified detections.
        attribution_data: List of attribution report entries.
        scene_id: Scene identifier.
        output_path: Output file path.

    Returns:
        Path to the saved CSV.
    """
    if len(detections_gdf) == 0:
        logger.info("Empty GeoDataFrame provided. Returning basic CSV with headers.")
        df = pd.DataFrame(columns=[
            "cluster_id", "lat", "lon", "area_sq_m", "polymer_type",
            "confidence", "top_source_type", "top_source_location",
            "top_source_country", "attribution_score", "detection_date",
            "scene_id",
        ])
        df.to_csv(output_path, index=False)
        return output_path

    attr_by_cluster = {}
    for attr in attribution_data:
        cid = attr.get("debris_cluster_id")
        if cid is not None:
            attr_by_cluster[cid] = attr

    rows = []
    for _, det in detections_gdf.iterrows():
        cid = det.get("cluster_id", 0)
        attr = attr_by_cluster.get(cid, {})

        rows.append({
            "cluster_id": cid,
            "lat": det.get("centroid_lat", 0),
            "lon": det.get("centroid_lon", 0),
            "area_sq_m": det.get("area_m2", 0),
            "polymer_type": det.get("polymer_type", ""),
            "confidence": det.get("mean_confidence", 0),
            "top_source_type": attr.get("source_type", ""),
            "top_source_location": attr.get("location_name", ""),
            "top_source_country": attr.get("country", ""),
            "attribution_score": attr.get("attribution_score", 0),
            "detection_date": det.get("detection_date", ""),
            "scene_id": scene_id,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info("CSV summary saved to %s", output_path)
    return output_path


# ─────────────────────────────────────────────
# TERMINAL SUMMARY
# ─────────────────────────────────────────────
def print_terminal_summary(
    scene_id: str,
    detections_gdf: gpd.GeoDataFrame,
    attribution_data: List[Dict[str, Any]],
):
    """Print a rich-formatted terminal summary table.

    Args:
        scene_id: Scene identifier.
        detections_gdf: GeoDataFrame of classified detections.
        attribution_data: List of attribution report entries.
    """
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()

        # Header
        console.print(
            Panel(
                f"[bold cyan]Plastic-Ledger Report[/] — Scene: [bold]{scene_id}[/]",
                style="cyan",
            )
        )

        n_clusters = len(detections_gdf)
        total_area = detections_gdf["area_m2"].sum() if n_clusters > 0 and "area_m2" in detections_gdf.columns else 0

        console.print(f"\n  [bold]Debris Clusters:[/] {n_clusters}")
        console.print(f"  [bold]Total Area:[/] {total_area:.0f} m² ({total_area/1e6:.4f} km²)")

        if n_clusters > 0:
            # Detections table
            table = Table(
                title="Detected Debris Clusters",
                show_lines=True,
                style="cyan",
            )
            table.add_column("ID", style="bold")
            table.add_column("Area (sq m)", justify="right")
            table.add_column("Confidence", justify="right")
            table.add_column("Polymer Type")
            table.add_column("Location")

            for _, row in detections_gdf.iterrows():
                table.add_row(
                    str(row.get("cluster_id", "")),
                    f"{row.get('area_m2', 0):.0f}",
                    f"{row.get('mean_confidence', 0):.3f}",
                    str(row.get("polymer_type", "N/A")),
                    f"({row.get('centroid_lon', 0):.3f}, "
                    f"{row.get('centroid_lat', 0):.3f})",
                )

            console.print(table)

        # Attribution table
        if attribution_data:
            attr_table = Table(
                title="Source Attribution",
                show_lines=True,
                style="green",
            )
            attr_table.add_column("Rank", style="bold")
            attr_table.add_column("Source Type")
            attr_table.add_column("Score", justify="right")
            attr_table.add_column("Location")
            attr_table.add_column("Explanation")

            for i, attr in enumerate(attribution_data[:5]):
                score = attr.get("attribution_score", 0)
                score_color = "green" if score > 0.6 else "yellow" if score > 0.3 else "red"
                attr_table.add_row(
                    f"#{i+1}",
                    attr.get("source_type", "Unknown"),
                    f"[{score_color}]{score*100:.0f}%[/]",
                    attr.get("location_name", "Unknown"),
                    attr.get("explanation", "")[:80] + "...",
                )

            console.print(attr_table)

    except ImportError:
        # Fallback without rich
        print(f"\n{'='*60}")
        print(f"  Plastic-Ledger Report — Scene: {scene_id}")
        print(f"{'='*60}")
        print(f"  Clusters: {len(detections_gdf)}")
        if attribution_data:
            for attr in attribution_data[:3]:
                print(
                    f"  #{attr.get('source_rank', '?')}: "
                    f"{attr.get('source_type', 'Unknown')} "
                    f"({attr.get('attribution_score', 0)*100:.0f}%)"
                )


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────
def run(
    scene_id: str,
    detections_path: Union[str, Path],
    attribution_path: Union[str, Path],
    output_dir: Union[str, Path] = "data/reports",
    config: Optional[Dict] = None,
) -> Dict[str, Path]:
    """Generate all report outputs for a scene.

    Args:
        scene_id: Scene identifier.
        detections_path: Path to classified detections GeoJSON.
        attribution_path: Path to attribution report JSON.
        output_dir: Root output directory.
        config: Optional config dict.

    Returns:
        Dict mapping output type → file path.
    """
    detections_path = Path(detections_path)
    attribution_path = Path(attribution_path)
    out_dir = Path(output_dir) / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    if stage_output_exists(out_dir, [
        "final_report.pdf", "final_report.geojson", "debris_summary.csv",
    ]):
        return {
            "pdf": out_dir / "final_report.pdf",
            "geojson": out_dir / "final_report.geojson",
            "csv": out_dir / "debris_summary.csv",
        }

    # Load data
    detections_gdf = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    if detections_path.exists():
        try:
            detections_gdf = gpd.read_file(detections_path)
        except Exception as exc:
            logger.warning("Could not load detections: %s", exc)

    attribution_data = []
    if attribution_path.exists():
        try:
            with open(attribution_path) as fh:
                attribution_data = json.load(fh)
        except Exception as exc:
            logger.warning("Could not load attribution: %s", exc)

    # Generate detection map
    detection_map = out_dir / "detection_map.png"
    _generate_detection_map(detections_gdf, detection_map)

    # Generate PDF
    pdf_path = generate_pdf(
        scene_id, detections_gdf, attribution_data, out_dir,
        detection_map_path=detection_map,
    )

    # Generate GeoJSON summary
    geojson_path = generate_geojson_summary(
        detections_gdf, attribution_data,
        out_dir / "final_report.geojson",
    )

    # Generate CSV
    csv_path = generate_csv(
        detections_gdf, attribution_data, scene_id,
        out_dir / "debris_summary.csv",
    )

    # Print terminal summary
    print_terminal_summary(scene_id, detections_gdf, attribution_data)

    logger.info(
        "[bold green]Stage 7 complete[/] — Reports in %s", out_dir,
    )

    return {
        "pdf": pdf_path,
        "geojson": geojson_path,
        "csv": csv_path,
    }


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    """CLI entrypoint for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Stage 7: Generate reports (PDF, GeoJSON, CSV)",
    )
    parser.add_argument("--scene_id", type=str, required=True)
    parser.add_argument("--detections", type=str, required=True)
    parser.add_argument("--attribution", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/reports")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    outputs = run(
        scene_id=args.scene_id,
        detections_path=args.detections,
        attribution_path=args.attribution,
        output_dir=args.output_dir,
        config=config,
    )
    print(f"\nReports generated:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

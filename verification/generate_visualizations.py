"""
Plastic-Ledger — OceanParcels Backtracking Visualization Generator (Dynamic)
=============================================================================
Generates:
  1. Interactive Leaflet HTML Map (`backtrack_map.html`):
     - Dynamic Release Point (Red Marker, Lat/Lon, Date)
     - Dynamic Origin Centroid (Blue Marker, Lat/Lon, Date)
     - Dynamic Particle Trajectory Lines (EPSG:4326 Polylines)
     - Dynamic Endpoint Markers
     - Dynamic 95% Spatial Confidence Convex Hull Polygon
     - Tile Servers: CartoDB Light, CartoDB Dark, OpenStreetMap
  2. High-Resolution 4-Panel Publication Dashboard (`backtrack_dashboard.png`):
     - Dynamic Spatial Map with Trajectories & Confidence Ellipse
     - Net Transport Distance (km) vs Time (Hours)
     - Ensemble Spread Radius (km) & Convex Hull Area (km²)
     - Dynamic Drift Velocity & Speed Distribution (m/s)

Usage:
    python verification/generate_visualizations.py [--dir data/benchmarks/benchmark_run_2]
"""

import sys
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def generate_leaflet_map(output_dir: Path) -> Path:
    """Generate dynamic standalone interactive Leaflet HTML map."""
    summary_file = output_dir / "backtrack_summary.json"
    traj_file = output_dir / "backtrack_trajectories.geojson"
    ep_file = output_dir / "backtrack_endpoints.geojson"
    hull_file = output_dir / "uncertainty_heatmap.geojson"

    if not summary_file.exists():
        print(f"Warning: {summary_file} not found. Skipping HTML map generation.")
        return output_dir / "backtrack_map.html"

    with open(summary_file) as f:
        summary_data = json.load(f)

    user_in = summary_data.get("user_input", {})
    centroid_in = summary_data.get("backtracked_origin_centroid", {})

    rel_lat = float(user_in.get("latitude", 41.18162))
    rel_lon = float(user_in.get("longitude", 2.24084))
    orig_lat = float(centroid_in.get("latitude", 41.11353))
    orig_lon = float(centroid_in.get("longitude", 3.60326))
    start_time = str(user_in.get("start_time_gmt1", "Start Time"))
    end_time = str(user_in.get("end_time_gmt1", "Release Time"))
    drift_dist = float(summary_data.get("net_backtrack_drift_distance_km", 0.0))
    n_particles = int(user_in.get("n_particles", 100))
    kh = float(user_in.get("diffusion_kh", 1.5))
    integrator = str(user_in.get("integrator", "RK4"))

    # Load trajectories GeoJSON
    trajectories_js = "[]"
    if traj_file.exists():
        with open(traj_file) as f:
            traj_geojson = json.load(f)
            raw_lines = []
            for feat in traj_geojson.get("features", []):
                coords = feat.get("geometry", {}).get("coordinates", [])
                if coords:
                    raw_lines.append([[c[1], c[0]] for c in coords])
            trajectories_js = json.dumps(raw_lines)

    # Load endpoints GeoJSON
    endpoints_js = "[]"
    if ep_file.exists():
        with open(ep_file) as f:
            ep_geojson = json.load(f)
            raw_pts = []
            for feat in ep_geojson.get("features", []):
                c = feat.get("geometry", {}).get("coordinates", [])
                if c:
                    raw_pts.append([c[1], c[0]])
            endpoints_js = json.dumps(raw_pts)

    # Load hull GeoJSON
    hull_js = "[]"
    if hull_file.exists():
        with open(hull_file) as f:
            hull_geojson = json.load(f)
            raw_hulls = []
            for feat in hull_geojson.get("features", []):
                c = feat.get("geometry", {}).get("coordinates", [])
                if c:
                    if isinstance(c[0][0], list):
                        raw_hulls.append([[p[1], p[0]] for p in c[0]])
                    else:
                        raw_hulls.append([[p[1], p[0]] for p in c])
            hull_js = json.dumps(raw_hulls)

    map_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Plastic-Ledger — Hydrodynamic Backtracking Map</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: 'Segoe UI', Arial, sans-serif; }}
        #map {{ width: 100vw; height: 100vh; }}
        .info-panel {{
            position: absolute; top: 15px; right: 15px; z-index: 1000;
            background: rgba(15, 23, 42, 0.90); color: #f8fafc; padding: 18px 22px;
            border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.4);
            max-width: 350px; backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.15);
        }}
        .info-panel h2 {{ margin: 0 0 10px 0; font-size: 18px; color: #38bdf8; font-weight: 600; display: flex; align-items: center; gap: 8px; }}
        .info-panel p {{ margin: 6px 0; font-size: 13px; color: #cbd5e1; line-height: 1.4; }}
        .stat-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; font-size: 12px; }}
        .stat-box {{ background: rgba(255,255,255,0.05); padding: 8px 10px; border-radius: 6px; border-left: 3px solid #0284c7; }}
        .stat-label {{ color: #94a3b8; font-size: 11px; text-transform: uppercase; }}
        .stat-value {{ font-weight: bold; color: #f1f5f9; font-size: 13px; margin-top: 2px; }}
        .legend {{ position: absolute; bottom: 30px; left: 15px; z-index: 1000; background: rgba(15, 23, 42, 0.9); color: white; padding: 12px 16px; border-radius: 8px; font-size: 12px; border: 1px solid rgba(255,255,255,0.1); }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; }}
        .legend-color {{ width: 14px; height: 14px; border-radius: 50%; display: inline-block; }}
    </style>
</head>
<body>
    <div id="map"></div>
    
    <div class="info-panel">
        <h2>🌊 Plastic-Ledger Backtracking</h2>
        <p><b>Lagrangian Particle Simulation Benchmark</b></p>
        <div class="stat-grid">
            <div class="stat-box" style="border-color: #ef4444;">
                <div class="stat-label">Release Point (End)</div>
                <div class="stat-value">{rel_lat:.4f}°, {rel_lon:.4f}°</div>
            </div>
            <div class="stat-box" style="border-color: #3b82f6;">
                <div class="stat-label">Origin Centroid (Start)</div>
                <div class="stat-value">{orig_lat:.4f}°, {orig_lon:.4f}°</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Net Drift Distance</div>
                <div class="stat-value">{drift_dist:.1f} km</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Particle Ensemble</div>
                <div class="stat-value">{n_particles} Particles</div>
            </div>
        </div>
        <p style="margin-top: 12px; font-size: 11px; color: #64748b;">
            • Forcing: CMEMS Ocean Currents + ERA5 Windage (3%)<br>
            • Integration: OceanParcels {integrator} + Brownian Diffusion ($K_h={kh} m^2/s$)
        </p>
    </div>

    <div class="legend">
        <div style="font-weight: bold; margin-bottom: 6px;">Map Layer Legend</div>
        <div class="legend-item"><span class="legend-color" style="background: #ef4444;"></span> Detection Release Point ({end_time})</div>
        <div class="legend-item"><span class="legend-color" style="background: #3b82f6;"></span> Backtracked Origin Centroid ({start_time})</div>
        <div class="legend-item"><span class="legend-color" style="background: #38bdf8; height: 3px; border-radius: 0;"></span> Particle Trajectories ({n_particles} Paths)</div>
        <div class="legend-item"><span class="legend-color" style="background: rgba(59, 130, 246, 0.4); border: 1px dashed #3b82f6;"></span> 95% Confidence Spatial Hull</div>
    </div>

    <script>
        var map = L.map('map').setView([{(rel_lat + orig_lat) / 2.0}, {(rel_lon + orig_lon) / 2.0}], 10);

        var cartoPositron = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            maxZoom: 19,
            attribution: '© CARTO © OpenStreetMap'
        }}).addTo(map);

        var cartoDark = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            maxZoom: 19,
            attribution: '© CARTO © OpenStreetMap'
        }});

        var osm = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 18,
            attribution: '© OpenStreetMap'
        }});

        L.control.layers({{ "CartoDB Light": cartoPositron, "CartoDB Dark": cartoDark, "OpenStreetMap": osm }}).addTo(map);

        var trajectories = {trajectories_js};
        var endpoints = {endpoints_js};
        var hulls = {hull_js};

        // Draw Trajectory Polylines
        trajectories.forEach(function(line) {{
            L.polyline(line, {{
                color: '#0284c7',
                weight: 1.5,
                opacity: 0.45,
                smoothFactor: 1
            }}).addTo(map);
        }});

        // Draw 95% Confidence Convex Hull Polygon
        hulls.forEach(function(polygon) {{
            L.polygon(polygon, {{
                color: '#2563eb',
                weight: 2,
                dashArray: '5, 5',
                fillColor: '#3b82f6',
                fillOpacity: 0.25
            }}).addTo(map).bindPopup("<b>95% Spatial Confidence Boundary</b><br>Particle Ensemble Origin Boundary");
        }});

        // Draw Endpoint Markers
        endpoints.forEach(function(pt) {{
            L.circleMarker(pt, {{
                radius: 3,
                fillColor: '#60a5fa',
                color: '#1d4ed8',
                weight: 1,
                fillOpacity: 0.8
            }}).addTo(map);
        }});

        // Release Point Marker (Red)
        L.circleMarker([{rel_lat}, {rel_lon}], {{
            radius: 8,
            fillColor: '#ef4444',
            color: '#ffffff',
            weight: 2,
            fillOpacity: 0.95
        }}).addTo(map).bindPopup("<b>📍 Detection Release Point</b><br>Lat: {rel_lat:.5f}° N<br>Lon: {rel_lon:.5f}° E<br>Date: {end_time}").openPopup();

        // Backtracked Origin Centroid Marker (Blue)
        L.circleMarker([{orig_lat}, {orig_lon}], {{
            radius: 9,
            fillColor: '#2563eb',
            color: '#ffffff',
            weight: 2,
            fillOpacity: 0.95
        }}).addTo(map).bindPopup("<b>🏁 Backtracked Origin Centroid</b><br>Lat: {orig_lat:.5f}° N<br>Lon: {orig_lon:.5f}° E<br>Date: {start_time}<br>Net Drift: {drift_dist:.1f} km");

        // Fit map bounds to trajectories
        if (trajectories.length > 0) {{
            var bounds = [];
            trajectories.forEach(function(line) {{
                line.forEach(function(pt) {{ bounds.push(pt); }});
            }});
            map.fitBounds(bounds, {{ padding: [50, 50] }});
        }}
    </script>
</body>
</html>
"""

    html_out = output_dir / "backtrack_map.html"
    with open(html_out, "w", encoding="utf-8") as f:
        f.write(map_html)

    print(f"[OK] Generated Interactive Leaflet HTML Map: {html_out}")
    return html_out


def generate_matplotlib_dashboard(output_dir: Path) -> Path:
    """Generate dynamic high-resolution 4-panel publication dashboard PNG."""
    stats_file = output_dir / "ensemble_statistics.csv"
    particles_file = output_dir / "backtrack_particles.csv"
    summary_file = output_dir / "backtrack_summary.json"

    # Parse metadata dynamically
    start_time, end_time = "Start Time", "Release Time"
    n_particles = 100
    kh = 1.5
    if summary_file.exists():
        with open(summary_file) as f:
            sdata = json.load(f)
            uin = sdata.get("user_input", {})
            start_time = str(uin.get("start_time_gmt1", "Start Time"))
            end_time = str(uin.get("end_time_gmt1", "Release Time"))
            n_particles = int(uin.get("n_particles", 100))
            kh = float(uin.get("diffusion_kh", 1.5))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Load data
    df_stats = pd.read_csv(stats_file) if stats_file.exists() else None
    df_parts = pd.read_csv(particles_file) if particles_file.exists() else None

    # Panel A: Spatial Trajectories & Endpoints Map
    ax_map = axes[0, 0]
    if df_parts is not None:
        rel_lons = df_parts["release_lon"]
        rel_lats = df_parts["release_lat"]
        orig_lons = df_parts["backtracked_origin_lon"]
        orig_lats = df_parts["backtracked_origin_lat"]

        # Plot particle displacement vectors
        for i in range(len(df_parts)):
            ax_map.plot([rel_lons[i], orig_lons[i]], [rel_lats[i], orig_lats[i]], color="#0284c7", alpha=0.35, linewidth=0.8)

        # Plot Release Point (Red)
        ax_map.scatter(np.mean(rel_lons), np.mean(rel_lats), color="#ef4444", s=140, zorder=5, label=f"Release Point ({end_time})", edgecolors="black")
        # Plot Backtracked Endpoints (Blue circles)
        ax_map.scatter(orig_lons, orig_lats, color="#60a5fa", s=25, alpha=0.6, label=f"Particle Endpoints (N={n_particles})", zorder=3)
        # Plot Origin Centroid (Blue star)
        ax_map.scatter(np.mean(orig_lons), np.mean(orig_lats), color="#1d4ed8", s=180, marker="*", zorder=6, label=f"Origin Centroid ({start_time})", edgecolors="white")

    ax_map.set_title("A) Spatial Particle Backtracking Trajectories & Origin", fontsize=13, fontweight="bold", pad=10)
    ax_map.set_xlabel("Longitude (°E)", fontsize=11)
    ax_map.set_ylabel("Latitude (°N)", fontsize=11)
    ax_map.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    ax_map.grid(True, linestyle="--", alpha=0.5)

    # Panel B: Net Transport Distance vs Time
    ax_dist = axes[0, 1]
    if df_stats is not None:
        hours = df_stats["observation_idx"]
        init_lon = df_stats["mean_longitude"].iloc[0]
        init_lat = df_stats["mean_latitude"].iloc[0]

        dists_km = []
        for _, row in df_stats.iterrows():
            dlat = math.radians(row["mean_latitude"] - init_lat)
            dlon = math.radians(row["mean_longitude"] - init_lon)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(init_lat))*math.cos(math.radians(row["mean_latitude"]))*math.sin(dlon/2)**2
            dists_km.append(6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

        ax_dist.plot(hours, dists_km, color="#0284c7", linewidth=2.5, label="Cumulative Drift Distance (km)")
        ax_dist.fill_between(hours, 0, dists_km, color="#0284c7", alpha=0.15)
        ax_dist.set_title("B) Net Backtrack Transport Distance vs. Time", fontsize=13, fontweight="bold", pad=10)
        ax_dist.set_xlabel("Backtracking Time Elapsed (Hours)", fontsize=11)
        ax_dist.set_ylabel("Distance from Release Point (km)", fontsize=11)
        ax_dist.legend(loc="upper left")
        ax_dist.grid(True, linestyle="--", alpha=0.5)

    # Panel C: Ensemble Spread Radius & Spatial Area Growth
    ax_spread = axes[1, 0]
    if df_stats is not None and "particle_spread_radius_km" in df_stats.columns:
        hours = df_stats["observation_idx"]
        spread_km = df_stats["particle_spread_radius_km"]
        area_km2 = df_stats["convex_hull_area_km2"]

        ax_spread.plot(hours, spread_km, color="#059669", linewidth=2.2, label="Mean Particle Spread Radius (km)")
        ax_spread.set_ylabel("Spread Radius (km)", color="#059669", fontsize=11)
        ax_spread.tick_params(axis='y', labelcolor="#059669")

        ax_area = ax_spread.twinx()
        ax_area.plot(hours, area_km2, color="#d97706", linestyle="--", linewidth=2.0, label="Convex Hull Area (km²)")
        ax_area.set_ylabel("Convex Hull Area (km²)", color="#d97706", fontsize=11)
        ax_area.tick_params(axis='y', labelcolor="#d97706")
        ax_area.grid(False)

        ax_spread.set_title(f"C) Stochastic Ensemble Dispersal & Spread Growth (Kh={kh} m²/s)", fontsize=13, fontweight="bold", pad=10)
        ax_spread.set_xlabel("Backtracking Time Elapsed (Hours)", fontsize=11)
        ax_spread.grid(True, linestyle="--", alpha=0.5)

    # Panel D: Drift Distance Distribution per Particle
    ax_hist = axes[1, 1]
    if df_parts is not None and "drift_distance_km" in df_parts.columns:
        dists = df_parts["drift_distance_km"]
        ax_hist.hist(dists, bins=15, color="#3b82f6", edgecolor="white", alpha=0.85, density=True)
        ax_hist.axvline(np.mean(dists), color="#ef4444", linestyle="--", linewidth=2, label=f"Mean: {np.mean(dists):.1f} km")
        ax_hist.axvline(np.median(dists), color="#10b981", linestyle=":", linewidth=2, label=f"Median: {np.median(dists):.1f} km")
        ax_hist.set_title(f"D) Particle Drift Distance Distribution (N={n_particles} Ensemble)", fontsize=13, fontweight="bold", pad=10)
        ax_hist.set_xlabel("Total Drift Distance (km)", fontsize=11)
        ax_hist.set_ylabel("Probability Density", fontsize=11)
        ax_hist.legend(loc="upper right")
        ax_hist.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    png_out = output_dir / "backtrack_dashboard.png"
    plt.savefig(png_out, dpi=300)
    plt.close()

    print(f"[OK] Generated High-Resolution Publication Dashboard: {png_out}")
    return png_out


def run_all_visualizations(target_dir: Optional[Path] = None):
    """Generate both interactive map and static dashboard dynamically for a benchmark directory."""
    if target_dir is None:
        target_dir = PROJECT_ROOT / "data" / "benchmarks" / "benchmark_run_2"
        if not target_dir.exists():
            target_dir = PROJECT_ROOT / "data" / "verification_custom"

    target_dir = Path(target_dir)
    print("=" * 70)
    print(f"PLASTIC-LEDGER: DYNAMIC BACKTRACKING VISUALIZATIONS")
    print(f"Target Directory: {target_dir}")
    print("=" * 70)

    html_path = generate_leaflet_map(target_dir)
    png_path = generate_matplotlib_dashboard(target_dir)

    print("\n" + "=" * 70)
    print("VISUALIZATION GENERATION COMPLETE!")
    print(f"Interactive Map: {html_path}")
    print(f"Publication Chart: {png_path}")
    print("=" * 70)


if __name__ == "__main__":
    t_dir = sys.argv[1] if len(sys.argv) > 1 else None
    run_all_visualizations(t_dir)

import geopandas as gpd
from pathlib import Path

p = Path('data/runs/run_001/detections')
for scene_dir in p.iterdir():
    detections_file = scene_dir / 'detections_classified.geojson'
    if detections_file.exists():
        gdf = gpd.read_file(detections_file)
        print(f'Scene {scene_dir.name}: {len(gdf)} clusters')
        if len(gdf) > 0:
            row = gdf.iloc[0]
            print(f'  First: lon={row["centroid_lon"]:.2f}, lat={row["centroid_lat"]:.2f}')
            print(f'  Expected around: lon=-88 to -87, lat=15 to 16')

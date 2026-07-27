import rasterio
from pathlib import Path
import numpy as np
from pyproj import Transformer

patches_dir = Path("ml_training/dataset/MARIDA/patches")
results = []

for scene_dir in patches_dir.iterdir():
    if not scene_dir.is_dir():
        continue
    
    # Skip Honduras 16PCC / 16PDC if desired, or examine other tiles like 48MYU, 19QDA, 50LLR, 36JUN, 51PTS, 18QYF, 16PEC, 16QED, 30VWH
    scene_name = scene_dir.name
    
    for tif in scene_dir.glob("*_cl.tif"):
        with rasterio.open(tif) as src:
            lbl = src.read(1)
            debris_count = int((lbl == 1).sum())  # Class 1 = Marine Debris
            if debris_count > 50:
                img_tif = tif.parent / tif.name.replace("_cl.tif", ".tif")
                if img_tif.exists():
                    with rasterio.open(img_tif) as img_src:
                        bounds = img_src.bounds
                        crs = img_src.crs
                        
                        # Convert bounds to EPSG:4326 (Lon/Lat)
                        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                        min_lon, min_lat = transformer.transform(bounds.left, bounds.bottom)
                        max_lon, max_lat = transformer.transform(bounds.right, bounds.top)
                        
                        # Extract date from scene_name (e.g., S2_1-12-19_48MYU -> 2019-12-01)
                        results.append({
                            "scene": scene_name,
                            "patch": tif.name,
                            "debris_pixels": debris_count,
                            "crs": str(crs),
                            "bbox_wgs84": [round(min_lon, 4), round(min_lat, 4), round(max_lon, 4), round(max_lat, 4)],
                            "bounds_native": [bounds.left, bounds.bottom, bounds.right, bounds.top]
                        })

# Sort by debris_pixels descending
results.sort(key=lambda x: x["debris_pixels"], reverse=True)

print(f"Found {len(results)} patches with significant Marine Debris (Class 1):")
for r in results[:10]:
    print(f"Scene: {r['scene']} | Patch: {r['patch']} | Debris Px: {r['debris_pixels']}")
    print(f"  BBOX (WGS84): {r['bbox_wgs84']}")
    print(f"  CRS: {r['crs']}")

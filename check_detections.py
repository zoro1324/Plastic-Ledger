import json
from pathlib import Path
import rasterio

# Check what was detected
det_dir = Path('data/runs/run_final_verify/detections')
if det_dir.exists():
    for geojson_file in det_dir.glob('*/detections*.geojson'):
        with open(geojson_file) as f:
            data = json.load(f)
        print(f'File: {geojson_file.name}')
        print(f'Features: {len(data.get("features", []))}')
        print(f'Type: {data.get("type")}')
        print()

# Check the probability maps
prob_dir = Path('data/runs/run_final_verify/detections')
for tif_file in prob_dir.glob('*/*debris_prob*.tif'):
    print(f'Found probability map: {tif_file}')
    with rasterio.open(tif_file) as src:
        data = src.read(1)
        print(f'  Min prob: {data.min():.4f}, Max prob: {data.max():.4f}')
        print(f'  Mean prob: {data.mean():.4f}')
        print(f'  % > 0.1: {(data > 0.1).sum() / data.size * 100:.2f}%')
        print(f'  % > 0.01: {(data > 0.01).sum() / data.size * 100:.2f}%')
        print(f'  % > 0.001: {(data > 0.001).sum() / data.size * 100:.2f}%')

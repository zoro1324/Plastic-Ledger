"""
MARIDA Path Diagnostic
======================
Run this to see exactly what's in your split files vs what's on disk.

Usage:
    python marida_debug.py --data_dir D:\Plastic-Ledger\models\dataset\MARIDA
"""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
args   = parser.parse_args()

data_dir = Path(args.data_dir)

print("\n" + "="*60)
print("  SPLIT FILE SAMPLES")
print("="*60)
splits_dir = data_dir / "splits"
for sf in sorted(splits_dir.glob("*.txt")):
    lines = sf.read_text().strip().splitlines()
    print(f"\n📄 {sf.name}  ({len(lines)} entries)")
    print("  First 5 lines:")
    for l in lines[:5]:
        print(f"    [{repr(l)}]")

print("\n" + "="*60)
print("  ACTUAL .TIF FILES ON DISK (first 10 image patches)")
print("="*60)
all_tifs = list(data_dir.rglob("*.tif"))
img_tifs = [f for f in all_tifs if "_cl" not in f.stem and "_conf" not in f.stem]
print(f"\n  Total .tif files found: {len(all_tifs)}")
print(f"  Image patches found   : {len(img_tifs)}")
print(f"\n  First 10 image patch stems:")
for f in sorted(img_tifs)[:10]:
    print(f"    stem : [{f.stem}]")
    print(f"    path : {f.relative_to(data_dir)}")

print("\n" + "="*60)
print("  DIRECTORY STRUCTURE (2 levels)")
print("="*60)
for p in sorted(data_dir.iterdir()):
    print(f"  {p.name}/")
    if p.is_dir():
        sub = sorted(p.iterdir())[:5]
        for s in sub:
            print(f"    {s.name}")
        if len(list(p.iterdir())) > 5:
            print(f"    ... ({len(list(p.iterdir()))} total)")
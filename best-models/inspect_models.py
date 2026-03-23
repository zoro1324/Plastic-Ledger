import torch
import os

def main():
    files = ['best_model_UNet.pth', 'best_model_resnet.pth', 'best_model_SegTransformer.pth']
    for f in files:
        if not os.path.exists(f):
            print(f"\n--- {f} --- (File not found!)")
            continue
        try:
            # Add weights_only=False locally to inspect older pickle dumps if needed
            sd = torch.load(f, map_location='cpu')
            print(f"\n--- {f} ---")
            print(f"Type: {type(sd)}")
            if isinstance(sd, dict):
                print(f"Number of keys: {len(sd)}")
                print(f"First 10 keys: {list(sd.keys())[:10]}")
        except Exception as e:
            print(f"\n--- {f} --- ERROR: {e}")

if __name__ == "__main__":
    main()

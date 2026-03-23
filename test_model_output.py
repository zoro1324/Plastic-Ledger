import torch
import numpy as np
from pathlib import Path
import sys

# Test loading and running the model
sys.path.insert(0, '/d/Plastic-Ledger/src')
from pipeline.utils.model_utils import load_model

# Load model
model_path = Path('best-models/best_model_SegTransformer.pth')
print(f"Loading model from: {model_path}")
print(f"Model exists: {model_path.exists()}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

try:
    model = load_model(model_path, device=device, num_classes=15, num_bands=11)
    print(f"✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    sys.exit(1)

# Check model state
print(f"\nModel state summary:")
print(f"Model type: {type(model)}")
print(f"Model device: {next(model.parameters()).device}")
print(f"Model training mode: {model.training}")

# Load a preprocessed patch to test
preprocess_dir = Path('data/runs/run_final_verify/preprocessing')
patch_files = list(preprocess_dir.glob('**/patch_*.bin'))
if patch_files:
    patch_file = patch_files[0]
    print(f"\nLoading test patch: {patch_file}")
    patch_data = np.load(patch_file)
    print(f"Patch shape: {patch_data.shape}")
    print(f"Patch dtype: {patch_data.dtype}")
    print(f"Patch min/max: {patch_data.min():.4f}/{patch_data.max():.4f}")
    print(f"Patch mean/std: {patch_data.mean():.4f}/{patch_data.std():.4f}")
    
    # Convert to tensor and run inference
    patch_tensor = torch.from_numpy(patch_data).unsqueeze(0).float().to(device)
    print(f"Tensor shape: {patch_tensor.shape}")
    print(f"Tensor range: {patch_tensor.min():.4f}/{patch_tensor.max():.4f}")
    
    with torch.no_grad():
        output = model(patch_tensor)
    print(f"\nModel output shape: {output.shape}")
    print(f"Output range: {output.min():.4f}/{output.max():.4f}")
    
    # Get debris class (class 0) probabilities
    debris_probs = torch.softmax(output, dim=1)[0, 0]  # class 0 = debris
    print(f"Debris class probabilities:")
    print(f"  Min: {debris_probs.min():.6f}, Max: {debris_probs.max():.6f}")
    print(f"  Mean: {debris_probs.mean():.6f}")
    print(f"  % > 0.5: {(debris_probs > 0.5).sum().item() / debris_probs.numel() * 100:.2f}%")
    print(f"  % > 0.1: {(debris_probs > 0.1).sum().item() / debris_probs.numel() * 100:.2f}%")
    
    # Check all classes
    all_probs = torch.softmax(output, dim=1)[0]
    print(f"\nAll class max probabilities:")
    for c in range(15):
        print(f"  Class {c:2d}: {all_probs[c].max():.6f}")
else:
    print("No preprocessed patches found")

"""
MARIDA Marine Debris Balance-and-Train Workflow
==============================================
Complete pipeline to oversample Marine Debris and retrain SegTransformer.

Usage:
    python run_balanced_training_workflow.py \
        --src_data D:/Plastic-Ledger/U-net-models/dataset/MARIDA \
        --balanced_data D:/Plastic-Ledger/U-net-models/dataset/MARIDA_BALANCED \
        --output runs/segtransformer_v4_balanced \
        --step all
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n{'='*80}")
    print(f"📍 {description}")
    print(f"{'='*80}\n")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        return False
    
    print(f"\n✅ Completed: {description}")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_data", type=str, required=True,
                        help="Path to original MARIDA dataset")
    parser.add_argument("--balanced_data", type=str, required=True,
                        help="Path to save balanced dataset")
    parser.add_argument("--output", type=str, default="runs/segtransformer_v4_balanced",
                        help="Output directory for trained model")
    parser.add_argument("--step", type=str, default="all",
                        choices=["analyze", "oversample", "train", "all"],
                        help="Which step(s) to run")
    parser.add_argument("--target_debris_ratio", type=float, default=0.35,
                        help="Target ratio of debris patches in training set")
    parser.add_argument("--loss", type=str, default="dice",
                        choices=["ce", "dice", "focal"],
                        help="Loss function to use")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"🚀 MARIDA MARINE DEBRIS BALANCED TRAINING WORKFLOW")
    print(f"{'='*80}\n")
    print(f"Source dataset:    {args.src_data}")
    print(f"Balanced dataset:   {args.balanced_data}")
    print(f"Output model:      {args.output}")
    print(f"Target debris ratio: {args.target_debris_ratio*100:.0f}%")
    print(f"Loss function:     {args.loss}")
    print(f"Epochs:            {args.epochs}")
    print(f"Batch size:        {args.batch_size}")
    
    # Step 1: Analyze imbalance
    if args.step in ["analyze", "all"]:
        cmd = f"python analyze_class_imbalance.py --data_dir \"{args.src_data}\" --output imbalance_report.json"
        if not run_command(cmd, "Step 1: Analyze Class Imbalance"):
            return
    
    # Step 2: Oversample
    if args.step in ["oversample", "all"]:
        cmd = f"python oversample_marine_debris.py --data_dir \"{args.src_data}\" --output_dir \"{args.balanced_data}\" --target_debris_ratio {args.target_debris_ratio}"
        if not run_command(cmd, "Step 2: Oversample Marine Debris"):
            return
    
    # Step 3: Train
    if args.step in ["train", "all"]:
        cmd = f"python train_segtransformer_balanced.py --data_dir \"{args.balanced_data}\" --output_dir \"{args.output}\" --batch_size {args.batch_size} --epochs {args.epochs} --lr {args.lr} --loss {args.loss}"
        if not run_command(cmd, "Step 3: Train SegTransformer on Balanced Data"):
            return
    
    print(f"\n{'='*80}")
    print(f"🎉 WORKFLOW COMPLETE!")
    print(f"{'='*80}\n")
    print(f"Next steps:")
    print(f"  1. Evaluate the model: python U-net-models/marida_evaluate.py \\")
    print(f"       --model {Path(args.output) / 'best_model.pth'} \\")
    print(f"       --data_dir {Path(args.src_data)} \\")
    print(f"       --split test --output evaluation/{Path(args.output).name} --tta")
    print(f"\n  2. Compare with previous models using best-models/evaluate_models.py")
    print()


if __name__ == "__main__":
    main()

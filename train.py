#!/usr/bin/env python3
"""
Main training entry point for SegFormer iris segmentation
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import training components
from training.train import main as train_main
from utils.config import parse_args, config_from_args, validate_config
import torch


def main():
    """Main entry point for training"""
    print("="*60)
    print("🔬 SEGFORMER IRIS SEGMENTATION TRAINING")
    print("📊 Based on Oracle's Comprehensive Analysis")
    print("="*60)
    
    # Check PyTorch and CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Parse arguments
    args = parse_args()
    
    # Create configuration
    config = config_from_args(args)
    
    # Validate configuration
    try:
        validate_config(config)
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return
    
    # Check dataset availability
    dataset_root = config['data']['dataset_root']
    if not os.path.exists(f"{dataset_root}/images") or not os.path.exists(f"{dataset_root}/masks"):
        print("❌ Dataset not found!")
        print("Expected structure:")
        print("  dataset/")
        print("    ├── images/")
        print("    └── masks/")
        return
    
    print(f"✅ Dataset found at {dataset_root}")
    
    # Start training
    try:
        train_main(config, use_wandb=not args.no_wandb)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

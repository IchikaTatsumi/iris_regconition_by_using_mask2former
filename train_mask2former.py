#!/usr/bin/env python3
"""
Training script for Mask2Former iris segmentation
"""

import sys
import os
from pathlib import Path
import json
import argparse

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
from torch.utils.data import DataLoader

from src.training.mask2former_trainer import Mask2FormerTrainer
from src.data.dataset import UbirisDataset


def create_dataloaders(config: dict) -> dict:
    """Create training and validation dataloaders"""
    data_config = config['data']
    
    # Create datasets
    train_dataset = UbirisDataset(
        dataset_root=data_config['dataset_root'],
        split='train',
        use_subject_split=data_config.get('use_subject_split', True),
        preserve_aspect=data_config.get('preserve_aspect', True),
        image_size=data_config.get('image_size', 512),
        seed=config.get('seed', 42)
    )
    
    val_dataset = UbirisDataset(
        dataset_root=data_config['dataset_root'],
        split='val',
        use_subject_split=data_config.get('use_subject_split', True),
        preserve_aspect=data_config.get('preserve_aspect', True),
        image_size=data_config.get('image_size', 512),
        seed=config.get('seed', 42)
    )
    
    # Custom collate function
    def collate_fn(batch):
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        result = {
            'pixel_values': pixel_values,
            'labels': labels
        }
        
        if 'boundary' in batch[0]:
            boundary = torch.stack([item['boundary'] for item in batch])
            result['boundary'] = boundary
        
        return result
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"Dataloaders created:")
    print(f"  Train: {len(train_loader)} batches, {len(train_dataset)} samples")
    print(f"  Val: {len(val_loader)} batches, {len(val_dataset)} samples")
    
    return {'train': train_loader, 'val': val_loader}


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train Mask2Former for iris segmentation')
    parser.add_argument('--config', type=str, default='configs/mask2former_config.json',
                       help='Path to config file')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🔬 MASK2FORMER IRIS SEGMENTATION TRAINING")
    print("📊 Query-based Transformer Segmentation")
    print("="*60)
    
    # Check PyTorch and CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return
    
    config = load_config(args.config)
    print(f"✅ Configuration loaded from {args.config}")
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Adjust batch size based on GPU memory if using CUDA
    if device.type == 'cuda':
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory_gb < 8 and config['data']['batch_size'] > 2:
            print(f"⚠️  GPU memory ({gpu_memory_gb:.1f}GB) < 8GB, reducing batch size to 2")
            config['data']['batch_size'] = 2
        elif gpu_memory_gb < 12 and config['data']['batch_size'] > 4:
            print(f"⚠️  GPU memory ({gpu_memory_gb:.1f}GB) < 12GB, reducing batch size to 4")
            config['data']['batch_size'] = 4
    
    # Check dataset
    dataset_root = config['data']['dataset_dir']
    images_dir = config['data']['images_dir']
    masks_dir = config['data']['masks_dir']
    
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print("❌ Dataset not found!")
        print("Expected structure:")
        print(f"  {images_dir}")
        print(f"  {masks_dir}")
        return
    
    print(f"✅ Dataset found at {dataset_root}")
    
    # Create dataloaders
    dataloaders = create_dataloaders(config)
    
    # Check for resume checkpoint
    resume_from = args.resume
    if resume_from is None:
        output_dir = Path(config.get('output_dir', 'outputs/mask2former_iris'))
        last_checkpoint = output_dir / 'checkpoints' / 'last.pt'
        if last_checkpoint.exists():
            print(f"🔄 Found existing checkpoint: {last_checkpoint}")
            resume_from = str(last_checkpoint)
    
    # Create trainer
    trainer = Mask2FormerTrainer(
        config=config,
        device=device,
        use_wandb=args.wandb,
        resume_from=resume_from
    )
    
    # Start training
    try:
        trainer.train(dataloaders['train'], dataloaders['val'])
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
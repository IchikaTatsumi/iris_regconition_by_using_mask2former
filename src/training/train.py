"""
Training orchestrator for SegFormer iris segmentation
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Any
from pathlib import Path

from .trainer import IrisSegmentationTrainer
from data.dataset import UbirisDataset


def create_dataloaders(config: Dict[str, Any]) -> Dict[str, DataLoader]:
    """Create training and validation dataloaders"""
    data_config = config['data']
    
    # Create datasets with enhanced features
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
    
    # Custom collate function for boundary maps
    def collate_fn(batch):
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        result = {
            'pixel_values': pixel_values,
            'labels': labels
        }
        
        # Add boundary if available
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


def main(config: Dict[str, Any], use_wandb: bool = True):
    """
    Main training function
    
    Args:
        config: Training configuration
        use_wandb: Whether to use Weights & Biases logging
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Adjust batch size based on GPU memory if using CUDA
    if device.type == 'cuda':
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory_gb < 8 and config['data']['batch_size'] > 4:
            print(f"⚠️  GPU memory ({gpu_memory_gb:.1f}GB) < 8GB, reducing batch size to 4")
            config['data']['batch_size'] = 4
        elif gpu_memory_gb < 12 and config['data']['batch_size'] > 8:
            print(f"⚠️  GPU memory ({gpu_memory_gb:.1f}GB) < 12GB, keeping batch size at 8")
            config['data']['batch_size'] = 8
    
    # Create dataloaders
    dataloaders = create_dataloaders(config)
    
    # Ensure output directory exists before checking for checkpoints
    output_dir = Path(config.get('output_dir', 'outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    print(f"📁 Output directory ensured: {output_dir}")
    
    # Check for existing checkpoint to resume from
    resume_from = None
    last_checkpoint = output_dir / 'checkpoints' / 'last.pt'
    if last_checkpoint.exists():
        print(f"🔄 Found existing checkpoint: {last_checkpoint}")
        resume_from = str(last_checkpoint)
    
    # Create trainer
    trainer = IrisSegmentationTrainer(
        config=config,
        device=device,
        use_wandb=use_wandb,
        resume_from=resume_from
    )
    
    # Start training
    trainer.train(dataloaders['train'], dataloaders['val'])

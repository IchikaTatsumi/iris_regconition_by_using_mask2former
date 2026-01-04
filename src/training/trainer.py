"""
SegFormer trainer for iris segmentation following Oracle recommendations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import os
import json
from typing import Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

from models import create_model
from losses import create_loss_function
from evaluation.metrics import IrisSegmentationMetrics, AverageMeter
from data.dataset import UbirisDataset
from utils.wandb_confusion_matrix import log_confusion_matrix_from_metrics, create_wandb_metrics_dashboard


class IrisSegmentationTrainer:
    """
    Trainer for SegFormer iris segmentation following Oracle's recommendations
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device = None,
        use_wandb: bool = True,
        resume_from: str = None
    ):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_wandb = use_wandb
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize loss function
        self.criterion = self._create_loss_function()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.patience_counter = 0
        
        # Metrics tracking
        self.train_metrics = IrisSegmentationMetrics(num_classes=config['num_classes'])
        self.val_metrics = IrisSegmentationMetrics(num_classes=config['num_classes'])
        
        # Loss tracking
        self.train_loss_meter = AverageMeter()
        self.val_loss_meter = AverageMeter()
        
        # Create output directories
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        
        # Load checkpoint if resuming
        if resume_from:
            self._load_checkpoint(resume_from)
        
        # Setup logging
        if self.use_wandb:
            self._setup_wandb()
    
    def _create_model(self) -> nn.Module:
        """Create SegFormer model with enhancements"""
        model_config = self.config['model']
        model = create_model(**model_config)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model created: {model_config.get('model_name', 'SegFormer')}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def _create_loss_function(self) -> nn.Module:
        """Create combined loss function with automatic class weighting"""
        loss_config = self.config['loss']
        
        # Try to load class weights from our utility
        class_weights = None
        weights_path = Path('class_weights.pt')
        
        if weights_path.exists():
            try:
                weights_info = torch.load(weights_path, map_location='cpu')
                class_weights = weights_info['weight_tensor'].to(self.device)
                print(f"✅ Loaded class weights: {class_weights}")
                print(f"   Class 0 (background): {class_weights[0]:.4f}")
                print(f"   Class 1 (iris): {class_weights[1]:.4f}")
            except Exception as e:
                print(f"⚠️  Failed to load class weights: {e}")
        else:
            print(f"⚠️  Class weights not found at {weights_path}")
            print("   Run: python class_weights_util.py")
        
        # Fallback class distribution if weights not available
        if 'class_distribution' in self.config:
            class_dist = torch.tensor(self.config['class_distribution'])
        else:
            # Updated distribution based on our analysis (background: 94.7%, iris: 5.3%)
            class_dist = torch.tensor([0.947, 0.053])
        
        criterion = create_loss_function(
            num_classes=self.config['num_classes'],
            class_distribution=class_dist,
            class_weights=class_weights,  # Pass the calculated weights
            device=self.device,
            **loss_config
        )
        
        return criterion
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create AdamW optimizer with proper weight decay"""
        opt_config = self.config['optimizer']
        
        # Separate parameters for different weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'norm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': decay_params, 'weight_decay': opt_config.get('weight_decay', 0.01)},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=opt_config.get('base_lr', 3e-5), **{k: v for k, v in opt_config.items() 
                                                  if k not in ['base_lr', 'weight_decay']})
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create polynomial learning rate scheduler"""
        scheduler_config = self.config.get('scheduler', {})
        
        if scheduler_config.get('type') == 'polynomial':
            total_steps = self.config['training']['num_epochs'] * self.config['training'].get('steps_per_epoch', 1000)
            warmup_steps = scheduler_config.get('warmup_steps', 1000)
            
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    return (1 - progress) ** scheduler_config.get('power', 0.9)
            
            scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            return scheduler
        
        return None
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging"""
        wandb.init(
            project=self.config.get('project_name', 'iris-segmentation'),
            config=self.config,
            name=self.config.get('run_name', 'segformer-iris'),
            tags=self.config.get('tags', ['segformer', 'iris', 'segmentation'])
        )
        
        # Watch model
        wandb.watch(self.model, log_freq=100)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint for resuming training"""
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            return
        
        print(f"🔄 Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_metric = checkpoint.get('best_metric', 0.0)
        
        print(f"✅ Resumed from epoch {self.current_epoch}, best metric: {self.best_metric:.4f}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.train_loss_meter.reset()
        self.train_metrics.reset()
        
        # Update epoch for model (unfreezing encoder if needed)
        if hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(self.current_epoch)
        
        # Update epoch for adaptive loss
        if hasattr(self.criterion, 'set_epoch'):
            self.criterion.set_epoch(self.current_epoch)
        
        progress_bar = tqdm(train_loader, desc=f'Train Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            boundary = batch.get('boundary', None)
            if boundary is not None:
                boundary = boundary.to(self.device)
            
            # Forward pass
            outputs = self.model(pixel_values, labels)
            
            # Prepare targets for loss computation
            targets = {'labels': labels}
            if boundary is not None:
                targets['boundary'] = boundary
            
            # Compute loss
            loss_dict = self.criterion(outputs, targets)
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            self.train_loss_meter.update(total_loss.item(), pixel_values.size(0))
            
            with torch.no_grad():
                predictions = torch.argmax(outputs['logits'], dim=1)
                self.train_metrics.update(predictions, labels)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'avg_loss': f'{self.train_loss_meter.avg:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 100 == 0:
                log_dict = {
                    'train/batch_loss': total_loss.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': self.current_epoch
                }
                log_dict.update({f'train/{k}': v.item() if hasattr(v, 'item') else v 
                               for k, v in loss_dict.items()})
                wandb.log(log_dict)
        
        # Compute epoch metrics
        epoch_metrics = self.train_metrics.compute_all_metrics()
        epoch_metrics['avg_loss'] = self.train_loss_meter.avg
        
        return epoch_metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        self.val_loss_meter.reset()
        self.val_metrics.reset()
        
        progress_bar = tqdm(val_loader, desc=f'Val Epoch {self.current_epoch}')
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # Move to device
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                boundary = batch.get('boundary', None)
                if boundary is not None:
                    boundary = boundary.to(self.device)
                
                # Forward pass
                outputs = self.model(pixel_values, labels)
                
                # Prepare targets for loss computation
                targets = {'labels': labels}
                if boundary is not None:
                    targets['boundary'] = boundary
                
                # Compute loss
                loss_dict = self.criterion(outputs, targets)
                total_loss = loss_dict['total_loss']
                
                # Update metrics
                self.val_loss_meter.update(total_loss.item(), pixel_values.size(0))
                
                predictions = torch.argmax(outputs['logits'], dim=1)
                boundary_preds = outputs.get('boundary_logits', None)
                self.val_metrics.update(predictions, labels, boundary_preds, boundary)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'avg_loss': f'{self.val_loss_meter.avg:.4f}'
                })
        
        # Compute epoch metrics
        epoch_metrics = self.val_metrics.compute_all_metrics()
        epoch_metrics['avg_loss'] = self.val_loss_meter.avg
        
        return epoch_metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop"""
        print(f"Starting training for {self.config['training']['num_epochs']} epochs...")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        for epoch in range(self.current_epoch, self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validate epoch
            val_metrics = self.validate_epoch(val_loader)
            
            # Log epoch results
            print(f"\nEpoch {epoch + 1}/{self.config['training']['num_epochs']}:")
            print(f"  Train Loss: {train_metrics['avg_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['avg_loss']:.4f}")
            print(f"  Val mIoU: {val_metrics['mean_iou']:.4f}")
            print(f"  Val Dice: {val_metrics['mean_dice']:.4f}")
            
            # Log to wandb
            if self.use_wandb:
                # Create comprehensive metrics dashboard
                create_wandb_metrics_dashboard(
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    epoch=epoch + 1
                )
                
                # Log confusion matrices every 10 epochs or at the end
                if (epoch + 1) % 10 == 0 or (epoch + 1) == self.config['training']['num_epochs']:
                    # Log validation confusion matrix
                    log_confusion_matrix_from_metrics(
                        metrics_obj=self.val_metrics,
                        epoch=epoch + 1,
                        phase="validation",
                        class_names=['Background/Pupil', 'Iris']
                    )
                    
                    # Log training confusion matrix
                    log_confusion_matrix_from_metrics(
                        metrics_obj=self.train_metrics,
                        epoch=epoch + 1,
                        phase="training",
                        class_names=['Background/Pupil', 'Iris']
                    )
            
            # Check for improvement
            current_metric = val_metrics['mean_iou']
            is_best = current_metric > self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                self.patience_counter = 0
                print(f"  🎯 New best validation mIoU: {self.best_metric:.4f}")
            else:
                self.patience_counter += 1
                print(f"  📈 No improvement for {self.patience_counter} epochs")
            
            # Save checkpoints
            self._save_checkpoint(epoch, is_best, val_metrics)
            
            # Early stopping
            if self.patience_counter >= self.config['training'].get('patience', 15):
                print(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
                break
        
        print(f"\nTraining completed! Best validation mIoU: {self.best_metric:.4f}")
    
    def _save_checkpoint(self, epoch: int, is_best: bool, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'metrics': metrics,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'checkpoints' / 'latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save last epoch checkpoint
        last_path = self.output_dir / 'checkpoints' / 'last.pt'
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'checkpoints' / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"  💾 Best model saved to {best_path}")
        
        # Save periodic checkpoints
        if (epoch + 1) % self.config['training'].get('save_freq', 50) == 0:
            epoch_path = self.output_dir / 'checkpoints' / f'epoch_{epoch+1}.pth'
            torch.save(checkpoint, epoch_path)


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


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SegFormer for iris segmentation')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Create dataloaders
    dataloaders = create_dataloaders(config)
    
    # Create trainer
    trainer = IrisSegmentationTrainer(
        config=config,
        device=device,
        use_wandb=args.wandb
    )
    
    # Start training
    trainer.train(dataloaders['train'], dataloaders['val'])


if __name__ == "__main__":
    main()

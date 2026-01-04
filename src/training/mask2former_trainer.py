"""
Mask2Former trainer for iris segmentation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import os
from typing import Dict, Any, Optional
import numpy as np
from pathlib import Path

from models import create_model
from src.losses.mask2former_loss import create_mask2former_loss
from evaluation.metrics import IrisSegmentationMetrics, AverageMeter


class Mask2FormerTrainer:
    """
    Trainer for Mask2Former iris segmentation
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
        """Create Mask2Former model"""
        model_config = self.config['model']
        
        # Use Mask2Former architecture
        model = create_model(
            architecture="mask2former",
            **model_config
        )
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model created: Mask2Former")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def _create_loss_function(self) -> nn.Module:
        """Create Mask2Former loss function"""
        loss_config = self.config.get('loss', {})
        
        # Try to load class weights
        class_weights = None
        weights_path = Path('class_weights.pt')
        
        if weights_path.exists():
            try:
                weights_info = torch.load(weights_path, map_location='cpu')
                class_weights = weights_info['weight_tensor'].to(self.device)
                print(f"✅ Loaded class weights: {class_weights}")
            except Exception as e:
                print(f"⚠️  Failed to load class weights: {e}")
        
        criterion = create_mask2former_loss(
            num_classes=self.config['num_classes'],
            class_weights=class_weights,
            device=self.device,
            **loss_config
        )
        
        return criterion
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with proper parameter groups"""
        opt_config = self.config['optimizer']
        
        # Separate parameters for different weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'norm' in name or 'layer_norm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': decay_params, 'weight_decay': opt_config.get('weight_decay', 0.01)},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=opt_config.get('base_lr', 1e-4), **{k: v for k, v in opt_config.items() 
                                                  if k not in ['base_lr', 'weight_decay']})
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
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
        elif scheduler_config.get('type') == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs'],
                eta_min=scheduler_config.get('min_lr', 1e-7)
            )
            return scheduler
        
        return None
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging"""
        wandb.init(
            project=self.config.get('project_name', 'iris-segmentation-mask2former'),
            config=self.config,
            name=self.config.get('run_name', 'mask2former-iris'),
            tags=self.config.get('tags', ['mask2former', 'iris', 'segmentation'])
        )
        
        wandb.watch(self.model, log_freq=100)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint for resuming training"""
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            return
        
        print(f"🔄 Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_metric = checkpoint.get('best_metric', 0.0)
        
        print(f"✅ Resumed from epoch {self.current_epoch}, best metric: {self.best_metric:.4f}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.train_loss_meter.reset()
        self.train_metrics.reset()
        
        # Update epoch for model
        if hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(self.current_epoch)
        
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
            
            # Prepare targets
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
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                boundary = batch.get('boundary', None)
                if boundary is not None:
                    boundary = boundary.to(self.device)
                
                # Forward pass
                outputs = self.model(pixel_values, labels)
                
                # Prepare targets
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
        print(f"Starting Mask2Former training for {self.config['training']['num_epochs']} epochs...")
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
                from utils.wandb_confusion_matrix import create_wandb_metrics_dashboard
                create_wandb_metrics_dashboard(
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    epoch=epoch + 1
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
        
        # Save latest
        latest_path = self.output_dir / 'checkpoints' / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save last
        last_path = self.output_dir / 'checkpoints' / 'last.pt'
        torch.save(checkpoint, last_path)
        
        # Save best
        if is_best:
            best_path = self.output_dir / 'checkpoints' / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"  💾 Best model saved to {best_path}")
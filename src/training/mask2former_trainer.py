"""
FIXED Mask2Former trainer for iris segmentation
Key fixes:
1. Import create_mask2former_loss instead of create_loss_function
2. Updated all function calls
3. Added better error handling
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

from src.models import create_model
from src.losses import create_mask2former_loss  
from src.evaluation.metrics import IrisSegmentationMetrics, AverageMeter


class Mask2FormerTrainer:
    """
    FIXED Trainer for Mask2Former iris segmentation
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
        
        print(f"🔧 Initializing Mask2Former Trainer...")
        print(f"📱 Device: {self.device}")
        
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
        self.train_metrics = IrisSegmentationMetrics(num_classes=config.get('num_classes', 2))
        self.val_metrics = IrisSegmentationMetrics(num_classes=config.get('num_classes', 2))
        
        # Loss tracking
        self.train_loss_meter = AverageMeter()
        self.val_loss_meter = AverageMeter()
        
        # Create output directories
        self.output_dir = Path(config.get('output_dir', 'outputs/mask2former_iris'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {self.output_dir}")
        
        # Load checkpoint if resuming
        if resume_from:
            self._load_checkpoint(resume_from)
        
        # Setup logging
        if self.use_wandb:
            self._setup_wandb()
    
    def _create_model(self) -> nn.Module:
        """FIXED: Create Mask2Former model"""
        model_config = self.config.get('model', {})
        
        # Ensure required parameters
        model_config.setdefault('architecture', 'mask2former')
        model_config.setdefault('num_labels', 2)
        model_config.setdefault('add_boundary_head', True)
        
        print(f"🏗️  Creating model with config:")
        print(f"   Architecture: {model_config.get('architecture')}")
        print(f"   Num labels: {model_config.get('num_labels')}")
        print(f"   Boundary head: {model_config.get('add_boundary_head')}")
        
        # Create model
        model = create_model(**model_config)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Model created:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        return model
    
    def _create_loss_function(self) -> nn.Module:
        """FIXED: Create loss function with correct import"""
        loss_config = self.config.get('loss', {})
        
        # Try to load class weights
        class_weights = None
        weights_path = Path('class_weights.pt')
        
        if weights_path.exists():
            try:
                weights_info = torch.load(weights_path, map_location='cpu', weights_only=False)
                class_weights = weights_info['weight_tensor'].to(self.device)
                print(f"✅ Loaded class weights: {class_weights}")
            except Exception as e:
                print(f"⚠️  Failed to load class weights: {e}")
        else:
            print(f"⚠️  No class weights file found at {weights_path}")
            # Use default weights from config if available
            if 'class_weights' in loss_config:
                class_weights = torch.tensor(
                    loss_config['class_weights'], 
                    dtype=torch.float32,
                    device=self.device
                )
                print(f"✅ Using config class weights: {class_weights}")
        
        # FIXED: Use create_mask2former_loss
        criterion = create_mask2former_loss(
            num_classes=self.config.get('num_classes', 2),
            class_weights=class_weights,
            device=self.device,
            ce_weight=loss_config.get('ce_weight', 0.5),
            dice_weight=loss_config.get('dice_weight', 0.5),
            boundary_weight=loss_config.get('boundary_weight', 0.25),
            use_focal=loss_config.get('use_focal', False)
        )
        
        print(f"✅ Loss function created")
        
        return criterion
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with proper parameter groups"""
        opt_config = self.config.get('optimizer', {})
        
        # Separate parameters for different weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'norm' in name or 'layer_norm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        base_lr = opt_config.get('base_lr', opt_config.get('learning_rate', 1e-4))
        weight_decay = opt_config.get('weight_decay', 0.01)
        
        optimizer = optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=base_lr)
        
        print(f"✅ Optimizer created: AdamW (lr={base_lr}, wd={weight_decay})")
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        scheduler_config = self.config.get('scheduler', {})
        training_config = self.config.get('training', {})
        
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'polynomial':
            total_steps = training_config.get('num_epochs', 160) * training_config.get('steps_per_epoch', 1000)
            warmup_steps = scheduler_config.get('warmup_steps', 1000)
            power = scheduler_config.get('power', 0.9)
            
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    return (1 - progress) ** power
            
            scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            print(f"✅ Scheduler: Polynomial (warmup={warmup_steps}, power={power})")
            return scheduler
            
        elif scheduler_type == 'cosine':
            T_max = training_config.get('num_epochs', 160)
            min_lr = scheduler_config.get('min_lr', 1e-7)
            
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=min_lr
            )
            print(f"✅ Scheduler: CosineAnnealing (T_max={T_max}, min_lr={min_lr})")
            return scheduler
        
        print(f"⚠️  No scheduler configured")
        return None
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging"""
        project_name = self.config.get('project_name', 'iris-segmentation-mask2former')
        run_name = self.config.get('run_name', 'mask2former-iris')
        tags = self.config.get('tags', ['mask2former', 'iris', 'segmentation'])
        
        wandb.init(
            project=project_name,
            config=self.config,
            name=run_name,
            tags=tags
        )
        
        wandb.watch(self.model, log_freq=100)
        print(f"✅ WandB initialized: {project_name}/{run_name}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint for resuming training"""
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            return
        
        print(f"🔄 Loading checkpoint from {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_metric = checkpoint.get('best_metric', 0.0)
            
            print(f"✅ Resumed from epoch {self.current_epoch}, best metric: {self.best_metric:.4f}")
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print(f"   Starting fresh training...")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.train_loss_meter.reset()
        self.train_metrics.reset()
        
        # Update epoch for model
        if hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(self.current_epoch)
        
        progress_bar = tqdm(train_loader, desc=f'Train Epoch {self.current_epoch + 1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
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
                
                # Check for NaN
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"\n⚠️  NaN/Inf loss detected at batch {batch_idx}")
                    continue
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
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
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    'grad': f'{grad_norm:.2f}'
                })
                
                # Log to wandb
                if self.use_wandb and batch_idx % 100 == 0:
                    log_dict = {
                        'train/batch_loss': total_loss.item(),
                        'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                        'train/grad_norm': grad_norm,
                        'epoch': self.current_epoch
                    }
                    log_dict.update({f'train/{k}': v.item() if hasattr(v, 'item') else v 
                                   for k, v in loss_dict.items() if k != 'total_loss'})
                    wandb.log(log_dict)
                    
            except Exception as e:
                print(f"\n❌ Error in batch {batch_idx}: {e}")
                continue
        
        # Compute epoch metrics
        epoch_metrics = self.train_metrics.compute_all_metrics()
        epoch_metrics['avg_loss'] = self.train_loss_meter.avg
        
        return epoch_metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        self.val_loss_meter.reset()
        self.val_metrics.reset()
        
        progress_bar = tqdm(val_loader, desc=f'Val Epoch {self.current_epoch + 1}')
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                try:
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
                    
                except Exception as e:
                    print(f"\n❌ Error in validation batch {batch_idx}: {e}")
                    continue
        
        # Compute epoch metrics
        epoch_metrics = self.val_metrics.compute_all_metrics()
        epoch_metrics['avg_loss'] = self.val_loss_meter.avg
        
        return epoch_metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop"""
        num_epochs = self.config.get('training', {}).get('num_epochs', 160)
        patience = self.config.get('training', {}).get('patience', 15)
        
        print(f"\n{'='*60}")
        print(f"🚀 STARTING MASK2FORMER TRAINING")
        print(f"{'='*60}")
        print(f"Epochs: {num_epochs}")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Patience: {patience}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validate epoch
            val_metrics = self.validate_epoch(val_loader)
            
            # Log epoch results
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs} Summary:")
            print(f"{'='*60}")
            print(f"  Train Loss: {train_metrics['avg_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['avg_loss']:.4f}")
            print(f"  Val mIoU: {val_metrics.get('mean_iou', 0):.4f}")
            print(f"  Val Dice: {val_metrics.get('mean_dice', 0):.4f}")
            if 'class_1_iou' in val_metrics:
                print(f"  Val Iris IoU: {val_metrics['class_1_iou']:.4f}")
            if 'boundary_f1' in val_metrics:
                print(f"  Boundary F1: {val_metrics.get('boundary_f1', 0):.4f}")
            
            # Log to wandb
            if self.use_wandb:
                try:
                    from utils.wandb_confusion_matrix import create_wandb_metrics_dashboard
                    create_wandb_metrics_dashboard(
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        epoch=epoch + 1
                    )
                except ImportError:
                    # Fallback logging
                    wandb.log({
                        'epoch': epoch + 1,
                        **{f'train/{k}': v for k, v in train_metrics.items()},
                        **{f'val/{k}': v for k, v in val_metrics.items()}
                    })
            
            # Check for improvement
            current_metric = val_metrics.get('mean_iou', 0)
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
            if self.patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered after {self.patience_counter} epochs without improvement")
                break
            
            print(f"{'='*60}\n")
        
        print(f"\n{'='*60}")
        print(f"✅ TRAINING COMPLETED!")
        print(f"{'='*60}")
        print(f"Best validation mIoU: {self.best_metric:.4f}")
        print(f"Total epochs: {self.current_epoch + 1}")
        print(f"{'='*60}\n")
    
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
        
        # Save periodic checkpoints
        save_freq = self.config.get('training', {}).get('save_frequency', 25)
        if (epoch + 1) % save_freq == 0:
            periodic_path = self.output_dir / 'checkpoints' / f'epoch_{epoch+1}.pt'
            torch.save(checkpoint, periodic_path)
            print(f"  💾 Checkpoint saved to {periodic_path}")
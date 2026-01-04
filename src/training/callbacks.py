"""
Training callbacks for monitoring and control
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import json


class EarlyStopping:
    """
    Early stopping callback to prevent overfitting
    """
    
    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = 'max',
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
        self.monitor_op = np.greater if mode == 'max' else np.less
        self.min_delta *= 1 if mode == 'max' else -1
    
    def __call__(self, current_score: float, model: torch.nn.Module) -> bool:
        """
        Check if training should stop
        
        Args:
            current_score: Current validation metric
            model: Model to potentially restore weights for
        
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_weights = model.state_dict().copy() if self.restore_best_weights else None
            return False
        
        if self.monitor_op(current_score, self.best_score + self.min_delta):
            self.best_score = current_score
            self.counter = 0
            self.best_weights = model.state_dict().copy() if self.restore_best_weights else None
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    print(f"Restored best model weights (score: {self.best_score:.4f})")
                return True
            return False


class ModelCheckpoint:
    """
    Model checkpoint callback
    """
    
    def __init__(
        self,
        save_dir: str,
        filename: str = 'best_model.pth',
        monitor: str = 'val_mean_iou',
        mode: str = 'max',
        save_best_only: bool = True,
        save_freq: int = 1
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        
        self.best_score = None
        self.monitor_op = np.greater if mode == 'max' else np.less
    
    def __call__(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, float],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        """
        Save model checkpoint if conditions are met
        
        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer state
            metrics: Current metrics
            scheduler: Learning rate scheduler (optional)
        """
        current_score = metrics.get(self.monitor, 0)
        
        # Check if this is the best model
        is_best = False
        if self.best_score is None or self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            is_best = True
        
        # Save checkpoint
        should_save = (not self.save_best_only) or is_best or (epoch % self.save_freq == 0)
        
        if should_save:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'best_score': self.best_score
            }
            
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            # Save with appropriate filename
            if is_best:
                save_path = self.save_dir / f'best_{self.filename}'
                torch.save(checkpoint, save_path)
                print(f"💾 Best model saved: {save_path} (score: {current_score:.4f})")
            
            if epoch % self.save_freq == 0:
                save_path = self.save_dir / f'epoch_{epoch}_{self.filename}'
                torch.save(checkpoint, save_path)
                print(f"💾 Checkpoint saved: {save_path}")


class LearningRateLogger:
    """
    Learning rate logging callback
    """
    
    def __init__(self, log_freq: int = 100):
        self.log_freq = log_freq
        self.lrs = []
        self.steps = []
        self.step_count = 0
    
    def __call__(self, optimizer: torch.optim.Optimizer) -> float:
        """
        Log current learning rate
        
        Args:
            optimizer: Current optimizer
        
        Returns:
            Current learning rate
        """
        current_lr = optimizer.param_groups[0]['lr']
        
        if self.step_count % self.log_freq == 0:
            self.lrs.append(current_lr)
            self.steps.append(self.step_count)
        
        self.step_count += 1
        return current_lr
    
    def plot_lr_schedule(self, save_path: Optional[str] = None):
        """Plot learning rate schedule"""
        import matplotlib.pyplot as plt
        
        if not self.lrs:
            print("No learning rate data to plot")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.steps, self.lrs)
        plt.xlabel('Training Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.yscale('log')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Learning rate plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class MetricsLogger:
    """
    Metrics logging and visualization callback
    """
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_metrics = {}
        self.val_metrics = {}
        self.epochs = []
    
    def log_epoch(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log metrics for an epoch"""
        self.epochs.append(epoch)
        
        for key, value in train_metrics.items():
            if key not in self.train_metrics:
                self.train_metrics[key] = []
            self.train_metrics[key].append(value)
        
        for key, value in val_metrics.items():
            if key not in self.val_metrics:
                self.val_metrics[key] = []
            self.val_metrics[key].append(value)
    
    def plot_metrics(self, metrics_to_plot: Optional[List[str]] = None):
        """Plot training metrics"""
        import matplotlib.pyplot as plt
        
        if metrics_to_plot is None:
            metrics_to_plot = ['avg_loss', 'mean_iou', 'mean_dice']
        
        available_metrics = set(self.train_metrics.keys()) & set(self.val_metrics.keys())
        metrics_to_plot = [m for m in metrics_to_plot if m in available_metrics]
        
        if not metrics_to_plot:
            print("No metrics available for plotting")
            return
        
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 5))
        if len(metrics_to_plot) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics_to_plot):
            axes[i].plot(self.epochs, self.train_metrics[metric], label=f'Train {metric}', marker='o')
            axes[i].plot(self.epochs, self.val_metrics[metric], label=f'Val {metric}', marker='s')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        save_path = self.log_dir / 'training_metrics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training metrics plot saved to {save_path}")
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        metrics_data = {
            'epochs': self.epochs,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
        
        save_path = self.log_dir / 'training_metrics.json'
        with open(save_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"Training metrics saved to {save_path}")


class ProgressCallback:
    """
    Progress monitoring callback
    """
    
    def __init__(self, print_freq: int = 10):
        self.print_freq = print_freq
        self.start_time = None
        self.epoch_start_time = None
    
    def on_train_start(self, total_epochs: int):
        """Called at the start of training"""
        import time
        self.start_time = time.time()
        print(f"🚀 Training started for {total_epochs} epochs")
    
    def on_epoch_start(self, epoch: int):
        """Called at the start of each epoch"""
        import time
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Called at the end of each epoch"""
        import time
        
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            
            if epoch % self.print_freq == 0 or epoch == 0:
                print(f"\nEpoch {epoch + 1} Summary:")
                print(f"  Time: {epoch_time:.1f}s")
                print(f"  Train Loss: {train_metrics.get('avg_loss', 0):.4f}")
                print(f"  Val Loss: {val_metrics.get('avg_loss', 0):.4f}")
                print(f"  Val mIoU: {val_metrics.get('mean_iou', 0):.4f}")
                print(f"  Val Dice: {val_metrics.get('mean_dice', 0):.4f}")
    
    def on_train_end(self, best_metric: float):
        """Called at the end of training"""
        import time
        
        if self.start_time is not None:
            total_time = time.time() - self.start_time
            hours = total_time // 3600
            minutes = (total_time % 3600) // 60
            
            print(f"\n🎉 Training completed!")
            print(f"  Total time: {int(hours)}h {int(minutes)}m")
            print(f"  Best validation mIoU: {best_metric:.4f}")

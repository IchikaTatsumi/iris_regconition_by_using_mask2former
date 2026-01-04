"""
Checkpoint management utilities
"""

import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import glob


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    save_path: Union[str, Path],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    additional_data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model checkpoint with all training state
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Current metrics
        config: Training configuration
        save_path: Path to save checkpoint
        scheduler: Learning rate scheduler (optional)
        additional_data: Additional data to save (optional)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if additional_data is not None:
        checkpoint.update(additional_data)
    
    # Ensure directory exists
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: torch.device = torch.device('cpu'),
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load model checkpoint and restore training state
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)  
        device: Device to load tensors to
        strict: Whether to strictly enforce state dict matching
    
    Returns:
        Checkpoint data (epoch, metrics, config, etc.)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            print("✅ Model state loaded successfully")
        except Exception as e:
            print(f"⚠️  Model state loading failed: {e}")
            if strict:
                raise
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✅ Optimizer state loaded successfully")
        except Exception as e:
            print(f"⚠️  Optimizer state loading failed: {e}")
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✅ Scheduler state loaded successfully")
        except Exception as e:
            print(f"⚠️  Scheduler state loading failed: {e}")
    
    return checkpoint


def find_best_checkpoint(checkpoint_dir: Union[str, Path], metric: str = 'mean_iou') -> Optional[Path]:
    """
    Find the best checkpoint based on a metric
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        metric: Metric to use for selection
    
    Returns:
        Path to best checkpoint or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    # Look for checkpoint files
    checkpoint_files = list(checkpoint_dir.glob('*.pth'))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return None
    
    best_checkpoint = None
    best_score = -float('inf')
    
    for checkpoint_file in checkpoint_files:
        try:
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            
            if 'metrics' in checkpoint and metric in checkpoint['metrics']:
                score = checkpoint['metrics'][metric]
                
                if score > best_score:
                    best_score = score
                    best_checkpoint = checkpoint_file
                    
        except Exception as e:
            print(f"Could not load checkpoint {checkpoint_file}: {e}")
            continue
    
    if best_checkpoint:
        print(f"Best checkpoint: {best_checkpoint} ({metric}: {best_score:.4f})")
        return best_checkpoint
    else:
        print(f"No valid checkpoint found with metric '{metric}'")
        return None


def resume_training(
    checkpoint_path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> int:
    """
    Resume training from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to restore
        optimizer: Optimizer to restore
        scheduler: Scheduler to restore (optional)
    
    Returns:
        Epoch to resume from
    """
    checkpoint = load_checkpoint(
        checkpoint_path, model, optimizer, scheduler, strict=False
    )
    
    resume_epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('metrics', {}).get('mean_iou', 0)
    
    print(f"Resuming training from epoch {resume_epoch}")
    print(f"Previous best mIoU: {best_metric:.4f}")
    
    return resume_epoch


def cleanup_old_checkpoints(
    checkpoint_dir: Union[str, Path],
    keep_last_n: int = 3,
    keep_best: bool = True
) -> None:
    """
    Clean up old checkpoint files to save disk space
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of latest checkpoints to keep
        keep_best: Whether to always keep the best checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return
    
    # Find all checkpoint files
    checkpoint_files = list(checkpoint_dir.glob('epoch_*.pth'))
    
    if len(checkpoint_files) <= keep_last_n:
        return
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Find best checkpoint if needed
    best_checkpoint = None
    if keep_best:
        best_checkpoint = find_best_checkpoint(checkpoint_dir)
    
    # Keep recent checkpoints and best
    to_keep = set(checkpoint_files[:keep_last_n])
    if best_checkpoint:
        to_keep.add(best_checkpoint)
    
    # Remove old checkpoints
    removed_count = 0
    for checkpoint_file in checkpoint_files:
        if checkpoint_file not in to_keep:
            try:
                checkpoint_file.unlink()
                removed_count += 1
            except Exception as e:
                print(f"Could not remove {checkpoint_file}: {e}")
    
    if removed_count > 0:
        print(f"Cleaned up {removed_count} old checkpoint files")


def export_model_for_inference(
    model: torch.nn.Module,
    checkpoint_path: Union[str, Path],
    export_path: Union[str, Path],
    input_size: tuple = (1, 3, 512, 512),
    export_format: str = 'torch'  # 'torch', 'onnx', 'torchscript'
) -> None:
    """
    Export trained model for inference
    
    Args:
        model: Model to export
        checkpoint_path: Path to trained checkpoint
        export_path: Path to save exported model
        input_size: Input tensor size for tracing
        export_format: Export format
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    
    if export_format == 'torch':
        # Save just the model state dict
        torch.save(model.state_dict(), export_path)
        print(f"Model exported to {export_path}")
        
    elif export_format == 'torchscript':
        # Trace model
        dummy_input = torch.randn(*input_size)
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(export_path)
        print(f"TorchScript model exported to {export_path}")
        
    elif export_format == 'onnx':
        # Export to ONNX
        dummy_input = torch.randn(*input_size)
        
        torch.onnx.export(
            model,
            dummy_input,
            export_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['pixel_values'],
            output_names=['logits'],
            dynamic_axes={
                'pixel_values': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            }
        )
        print(f"ONNX model exported to {export_path}")
    
    else:
        raise ValueError(f"Unsupported export format: {export_format}")
    
    # Save metadata
    metadata = {
        'model_type': 'iris_segmentation',
        'input_size': input_size,
        'num_classes': 2,
        'class_names': ['background/pupil', 'iris'],
        'export_format': export_format,
        'checkpoint_metrics': checkpoint.get('metrics', {}),
        'config': checkpoint.get('config', {})
    }
    
    metadata_path = export_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model metadata saved to {metadata_path}")

"""
WandB confusion matrix visualization utilities for iris segmentation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Union
import wandb
from sklearn.metrics import confusion_matrix
from pathlib import Path


def create_wandb_confusion_matrix(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    class_names: List[str] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    log_to_wandb: bool = True,
    save_local: bool = False,
    save_path: Optional[str] = None,
    step: Optional[int] = None
) -> plt.Figure:
    """
    Create and log confusion matrix to WandB
    
    Args:
        predictions: Predicted class labels [N] or [B, H, W]
        targets: Ground truth labels [N] or [B, H, W]
        class_names: Names of classes (default: ['Background/Pupil', 'Iris'])
        title: Title for the confusion matrix
        normalize: Whether to normalize the matrix (True for percentages)
        log_to_wandb: Whether to log to WandB
        save_local: Whether to save locally
        save_path: Local save path (if save_local=True)
        step: Step number for WandB logging
    
    Returns:
        matplotlib Figure object
    """
    # Set default class names for iris segmentation
    if class_names is None:
        class_names = ['Background/Pupil', 'Iris']
    
    # Convert tensors to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Flatten arrays if they're multi-dimensional
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Remove any invalid indices (e.g., ignore_index = -1)
    valid_mask = targets >= 0
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]
    
    # Compute confusion matrix
    cm = confusion_matrix(
        targets, 
        predictions, 
        labels=range(len(class_names))
    )
    
    # Create figure
    plt.style.use('default')  # Ensure clean style
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Normalize if requested
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = '.2%'
        cbar_label = 'Percentage'
    else:
        cm_display = cm
        fmt = 'd'
        cbar_label = 'Count'
    
    # Create heatmap
    im = ax.imshow(cm_display, interpolation='nearest', cmap='Blues')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")
    
    # Add text annotations
    thresh = cm_display.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                text = f'{cm_normalized[i, j]:.1%}\n({cm[i, j]})'
            else:
                text = f'{cm[i, j]}'
            
            ax.text(j, i, text, ha="center", va="center",
                   color="white" if cm_display[i, j] > thresh else "black",
                   fontsize=12, fontweight='bold')
    
    # Set labels and title
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add grid
    ax.set_xlim(-0.5, len(class_names) - 0.5)
    ax.set_ylim(-0.5, len(class_names) - 0.5)
    
    plt.tight_layout()
    
    # Log to WandB
    if log_to_wandb:
        try:
            # Log as wandb.Image
            wandb_image = wandb.Image(
                fig, 
                caption=f"{title} - Normalized: {normalize}"
            )
            
            # Create log dict
            log_dict = {"confusion_matrix": wandb_image}
            if step is not None:
                log_dict["step"] = step
            
            wandb.log(log_dict, step=step)
            
            # Also log raw confusion matrix data
            cm_table = wandb.Table(
                columns=["True", "Predicted", "Count", "Percentage"],
                data=[
                    [class_names[i], class_names[j], int(cm[i, j]), 
                     cm[i, j] / cm.sum() * 100]
                    for i in range(len(class_names))
                    for j in range(len(class_names))
                ]
            )
            wandb.log({"confusion_matrix_data": cm_table}, step=step)
            
        except Exception as e:
            print(f"Warning: Could not log confusion matrix to WandB: {e}")
    
    # Save locally if requested
    if save_local:
        if save_path is None:
            save_path = f"confusion_matrix_{title.lower().replace(' ', '_')}.png"
        
        fig.savefig(save_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Confusion matrix saved locally to: {save_path}")
    
    return fig


def create_wandb_classification_report(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    class_names: List[str] = None,
    log_to_wandb: bool = True,
    step: Optional[int] = None
) -> Dict[str, float]:
    """
    Create detailed classification report and log to WandB
    
    Args:
        predictions: Predicted class labels
        targets: Ground truth labels
        class_names: Names of classes
        log_to_wandb: Whether to log to WandB
        step: Step number for WandB logging
    
    Returns:
        Dictionary of classification metrics
    """
    from sklearn.metrics import classification_report, precision_recall_fscore_support
    
    # Set default class names
    if class_names is None:
        class_names = ['Background/Pupil', 'Iris']
    
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy().flatten()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy().flatten()
    
    # Remove invalid indices
    valid_mask = targets >= 0
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]
    
    # Compute detailed metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )
    
    # Create metrics dictionary
    metrics = {}
    
    # Per-class metrics
    for i, class_name in enumerate(class_names):
        if i < len(precision):
            metrics[f"{class_name}_precision"] = float(precision[i])
            metrics[f"{class_name}_recall"] = float(recall[i])
            metrics[f"{class_name}_f1"] = float(f1[i])
            metrics[f"{class_name}_support"] = int(support[i])
    
    # Macro averages
    metrics["macro_precision"] = float(np.mean(precision))
    metrics["macro_recall"] = float(np.mean(recall))
    metrics["macro_f1"] = float(np.mean(f1))
    
    # Weighted averages
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        targets, predictions, average='weighted', zero_division=0
    )
    metrics["weighted_precision"] = float(precision_w)
    metrics["weighted_recall"] = float(recall_w)
    metrics["weighted_f1"] = float(f1_w)
    
    # Overall accuracy
    accuracy = (predictions == targets).mean()
    metrics["accuracy"] = float(accuracy)
    
    # Log to WandB
    if log_to_wandb:
        try:
            # Create table for detailed view
            report_table = wandb.Table(
                columns=["Class", "Precision", "Recall", "F1-Score", "Support"],
                data=[
                    [class_names[i], 
                     precision[i] if i < len(precision) else 0,
                     recall[i] if i < len(recall) else 0,
                     f1[i] if i < len(f1) else 0,
                     support[i] if i < len(support) else 0]
                    for i in range(len(class_names))
                ]
            )
            
            log_dict = {
                "classification_report": report_table,
                **{f"classification/{k}": v for k, v in metrics.items()}
            }
            
            wandb.log(log_dict, step=step)
            
        except Exception as e:
            print(f"Warning: Could not log classification report to WandB: {e}")
    
    return metrics


def log_confusion_matrix_from_metrics(
    metrics_obj,
    epoch: int,
    phase: str = "validation",
    class_names: List[str] = None
) -> None:
    """
    Log confusion matrix from IrisSegmentationMetrics object
    
    Args:
        metrics_obj: IrisSegmentationMetrics instance with accumulated predictions
        epoch: Current epoch number
        phase: Phase name (train/validation)
        class_names: Names of classes
    """
    if not metrics_obj.predictions or not metrics_obj.targets:
        print("Warning: No predictions available in metrics object")
        return
    
    # Get accumulated predictions and targets
    predictions = np.concatenate(metrics_obj.predictions)
    targets = np.concatenate(metrics_obj.targets)
    
    # Create and log confusion matrix
    title = f"{phase.title()} Confusion Matrix - Epoch {epoch}"
    
    fig = create_wandb_confusion_matrix(
        predictions=predictions,
        targets=targets,
        class_names=class_names,
        title=title,
        normalize=True,
        log_to_wandb=True,
        step=epoch
    )
    
    # Close figure to save memory
    plt.close(fig)
    
    # Also create classification report
    create_wandb_classification_report(
        predictions=predictions,
        targets=targets,
        class_names=class_names,
        log_to_wandb=True,
        step=epoch
    )


def create_wandb_metrics_dashboard(
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    epoch: int,
    confusion_matrices: bool = True
) -> None:
    """
    Create comprehensive metrics dashboard for WandB
    
    Args:
        train_metrics: Training metrics dictionary
        val_metrics: Validation metrics dictionary
        epoch: Current epoch
        confusion_matrices: Whether to create confusion matrices
    """
    # Log basic metrics with proper prefixes
    log_dict = {
        "epoch": epoch,
        **{f"train/{k}": v for k, v in train_metrics.items()},
        **{f"val/{k}": v for k, v in val_metrics.items()}
    }
    
    # Add target comparison
    val_miou = val_metrics.get('mean_iou', 0)
    val_dice = val_metrics.get('mean_dice', 0)
    
    log_dict.update({
        "targets/miou_target": 0.90,
        "targets/dice_target": 0.93,
        "targets/miou_progress": val_miou / 0.90,
        "targets/dice_progress": val_dice / 0.93,
        "targets/meets_miou": val_miou >= 0.90,
        "targets/meets_dice": val_dice >= 0.93,
        "targets/meets_both": (val_miou >= 0.90) and (val_dice >= 0.93)
    })
    
    wandb.log(log_dict, step=epoch)


if __name__ == "__main__":
    # Test the confusion matrix function
    print("Testing WandB confusion matrix visualization...")
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate realistic iris segmentation predictions
    # Background class (0) is more common (~90%), Iris class (1) is less common (~10%)
    true_labels = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    
    # Add some prediction errors
    predicted_labels = true_labels.copy()
    error_mask = np.random.random(n_samples) < 0.05  # 5% error rate
    predicted_labels[error_mask] = 1 - predicted_labels[error_mask]  # Flip labels
    
    # Test without WandB logging
    print("Creating confusion matrix visualization...")
    fig = create_wandb_confusion_matrix(
        predictions=predicted_labels,
        targets=true_labels,
        class_names=['Background/Pupil', 'Iris'],
        title="Test Confusion Matrix",
        normalize=True,
        log_to_wandb=False,  # Don't log during testing
        save_local=True,
        save_path="test_confusion_matrix.png"
    )
    
    # Test classification report
    print("Creating classification report...")
    metrics = create_wandb_classification_report(
        predictions=predicted_labels,
        targets=true_labels,
        class_names=['Background/Pupil', 'Iris'],
        log_to_wandb=False  # Don't log during testing
    )
    
    print("Classification metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    plt.close(fig)
    print("Test completed successfully!")

"""
Comprehensive evaluation metrics for iris segmentation
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from scipy.spatial.distance import directed_hausdorff
import cv2
from typing import Dict, List, Tuple, Optional


class IrisSegmentationMetrics:
    """
    Comprehensive metrics for iris segmentation evaluation
    """
    
    def __init__(self, num_classes: int = 2, ignore_index: int = -1):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics"""
        self.predictions = []
        self.targets = []
        self.boundary_predictions = []
        self.boundary_targets = []
    
    def update(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        boundary_predictions: Optional[torch.Tensor] = None,
        boundary_targets: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with new batch
        
        Args:
            predictions: [B, C, H, W] logits or [B, H, W] class predictions
            targets: [B, H, W] ground truth
            boundary_predictions: [B, 1, H, W] boundary logits (optional)
            boundary_targets: [B, H, W] boundary ground truth (optional)
        """
        # Convert logits to predictions if needed
        if predictions.dim() == 4 and predictions.shape[1] > 1:
            predictions = torch.argmax(predictions, dim=1)
        elif predictions.dim() == 4:
            predictions = predictions.squeeze(1)
        
        # Store predictions and targets
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        
        # Store boundary predictions if provided
        if boundary_predictions is not None and boundary_targets is not None:
            if boundary_predictions.dim() == 4:
                boundary_predictions = torch.sigmoid(boundary_predictions).squeeze(1)
            
            self.boundary_predictions.extend(boundary_predictions.cpu().numpy())
            self.boundary_targets.extend(boundary_targets.cpu().numpy())
    
    def compute_pixel_accuracy(self) -> float:
        """Compute overall pixel accuracy"""
        predictions = np.concatenate(self.predictions)
        targets = np.concatenate(self.targets)
        
        valid_mask = targets != self.ignore_index
        correct = (predictions[valid_mask] == targets[valid_mask]).sum()
        total = valid_mask.sum()
        
        return correct / total if total > 0 else 0.0
    
    def compute_class_accuracy(self) -> Dict[str, float]:
        """Compute per-class accuracy"""
        predictions = np.concatenate(self.predictions)
        targets = np.concatenate(self.targets)
        
        accuracies = {}
        for class_id in range(self.num_classes):
            class_mask = (targets == class_id)
            if class_mask.sum() > 0:
                correct = (predictions[class_mask] == class_id).sum()
                accuracies[f'class_{class_id}_acc'] = correct / class_mask.sum()
            else:
                accuracies[f'class_{class_id}_acc'] = 0.0
        
        return accuracies
    
    def compute_iou(self) -> Dict[str, float]:
        """Compute IoU for each class and mean IoU"""
        predictions = np.concatenate(self.predictions)
        targets = np.concatenate(self.targets)
        
        ious = {}
        valid_ious = []
        
        for class_id in range(self.num_classes):
            pred_mask = (predictions == class_id)
            target_mask = (targets == class_id)
            
            intersection = (pred_mask & target_mask).sum()
            union = (pred_mask | target_mask).sum()
            
            if union > 0:
                iou = intersection / union
                ious[f'class_{class_id}_iou'] = iou
                valid_ious.append(iou)
            else:
                ious[f'class_{class_id}_iou'] = 0.0
        
        ious['mean_iou'] = np.mean(valid_ious) if valid_ious else 0.0
        
        return ious
    
    def compute_dice(self) -> Dict[str, float]:
        """Compute Dice coefficient for each class"""
        predictions = np.concatenate(self.predictions)
        targets = np.concatenate(self.targets)
        
        dice_scores = {}
        valid_dice = []
        
        for class_id in range(self.num_classes):
            pred_mask = (predictions == class_id)
            target_mask = (targets == class_id)
            
            intersection = (pred_mask & target_mask).sum()
            total = pred_mask.sum() + target_mask.sum()
            
            if total > 0:
                dice = (2 * intersection) / total
                dice_scores[f'class_{class_id}_dice'] = dice
                valid_dice.append(dice)
            else:
                dice_scores[f'class_{class_id}_dice'] = 0.0
        
        dice_scores['mean_dice'] = np.mean(valid_dice) if valid_dice else 0.0
        
        return dice_scores
    
    def compute_precision_recall_f1(self) -> Dict[str, float]:
        """Compute precision, recall, and F1 for each class"""
        predictions = np.concatenate(self.predictions)
        targets = np.concatenate(self.targets)
        
        # Remove ignore_index
        valid_mask = targets != self.ignore_index
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        metrics = {}
        for class_id in range(self.num_classes):
            if class_id < len(precision):
                metrics[f'class_{class_id}_precision'] = precision[class_id]
                metrics[f'class_{class_id}_recall'] = recall[class_id]
                metrics[f'class_{class_id}_f1'] = f1[class_id]
            else:
                metrics[f'class_{class_id}_precision'] = 0.0
                metrics[f'class_{class_id}_recall'] = 0.0
                metrics[f'class_{class_id}_f1'] = 0.0
        
        # Macro averages
        metrics['macro_precision'] = np.mean(precision)
        metrics['macro_recall'] = np.mean(recall)
        metrics['macro_f1'] = np.mean(f1)
        
        return metrics
    
    def compute_boundary_f1(self, tolerance: int = 2) -> float:
        """
        Compute Boundary F1 score with tolerance
        
        Args:
            tolerance: Pixel tolerance for boundary matching
        
        Returns:
            Boundary F1 score
        """
        if not self.boundary_predictions or not self.boundary_targets:
            return 0.0
        
        boundary_f1_scores = []
        
        for pred, target in zip(self.boundary_predictions, self.boundary_targets):
            # Threshold boundary predictions
            pred_binary = (pred > 0.5).astype(np.uint8)
            target_binary = target.astype(np.uint8)
            
            # Dilate target boundary for tolerance
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerance*2+1, tolerance*2+1))
            target_dilated = cv2.dilate(target_binary, kernel, iterations=1)
            
            # Compute precision and recall
            true_positives = (pred_binary & target_dilated).sum()
            predicted_positives = pred_binary.sum()
            actual_positives = target_binary.sum()
            
            precision = true_positives / predicted_positives if predicted_positives > 0 else 0
            recall = true_positives / actual_positives if actual_positives > 0 else 0
            
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            boundary_f1_scores.append(f1)
        
        return np.mean(boundary_f1_scores)
    
    def compute_hausdorff_distance(self) -> float:
        """
        Compute average Hausdorff distance for iris boundaries
        """
        if len(self.predictions) == 0:
            return float('inf')
        
        hausdorff_distances = []
        
        for pred, target in zip(self.predictions, self.targets):
            # Extract iris boundaries (class 1)
            pred_iris = (pred == 1).astype(np.uint8)
            target_iris = (target == 1).astype(np.uint8)
            
            # Find contours
            pred_contours, _ = cv2.findContours(pred_iris, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            target_contours, _ = cv2.findContours(target_iris, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if pred_contours and target_contours:
                # Get largest contours (main iris boundary)
                pred_contour = max(pred_contours, key=cv2.contourArea)
                target_contour = max(target_contours, key=cv2.contourArea)
                
                # Convert to point sets
                pred_points = pred_contour.reshape(-1, 2)
                target_points = target_contour.reshape(-1, 2)
                
                # Compute Hausdorff distance
                try:
                    hd = max(
                        directed_hausdorff(pred_points, target_points)[0],
                        directed_hausdorff(target_points, pred_points)[0]
                    )
                    hausdorff_distances.append(hd)
                except:
                    # Skip if contour is too small or other issues
                    continue
        
        return np.mean(hausdorff_distances) if hausdorff_distances else float('inf')
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all available metrics"""
        if len(self.predictions) == 0:
            return {}
        
        metrics = {}
        
        # Basic metrics
        metrics['pixel_accuracy'] = self.compute_pixel_accuracy()
        metrics.update(self.compute_class_accuracy())
        metrics.update(self.compute_iou())
        metrics.update(self.compute_dice())
        metrics.update(self.compute_precision_recall_f1())
        
        # Boundary metrics
        if self.boundary_predictions and self.boundary_targets:
            metrics['boundary_f1'] = self.compute_boundary_f1()
        
        # Hausdorff distance (expensive, computed less frequently)
        # metrics['hausdorff_distance'] = self.compute_hausdorff_distance()
        
        return metrics


def compute_confusion_matrix(predictions: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Compute confusion matrix
    
    Args:
        predictions: Predicted class labels
        targets: Ground truth labels
        num_classes: Number of classes
    
    Returns:
        Confusion matrix
    """
    return confusion_matrix(
        targets.flatten(), 
        predictions.flatten(), 
        labels=range(num_classes)
    )


def compute_class_weights_from_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    """
    Compute class weights from confusion matrix
    
    Args:
        cm: Confusion matrix
    
    Returns:
        Class weights
    """
    class_frequencies = cm.sum(axis=1)
    total_samples = class_frequencies.sum()
    num_classes = len(class_frequencies)
    
    weights = total_samples / (num_classes * class_frequencies)
    return weights


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def benchmark_inference_speed(
    model: torch.nn.Module, 
    input_size: Tuple[int, int, int, int] = (1, 3, 512, 512),
    device: torch.device = torch.device('cpu'),
    num_runs: int = 100,
    warmup_runs: int = 10
) -> Dict[str, float]:
    """
    Benchmark model inference speed
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
        device: Device to run on
        num_runs: Number of timing runs
        warmup_runs: Number of warmup runs
    
    Returns:
        Dictionary with timing statistics
    """
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(*input_size, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input, return_boundary=False)
    
    # Synchronize GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Timing runs
    import time
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input, return_boundary=False)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
    
    times = np.array(times)
    
    return {
        'mean_time': times.mean(),
        'std_time': times.std(),
        'min_time': times.min(),
        'max_time': times.max(),
        'fps': 1.0 / times.mean(),
        'throughput': input_size[0] / times.mean()  # samples per second
    }


if __name__ == "__main__":
    # Test metrics computation
    print("Testing iris segmentation metrics...")
    
    # Create dummy data
    batch_size = 4
    height, width = 128, 128
    num_classes = 2
    
    # Simulate predictions and targets
    predictions = torch.randint(0, num_classes, (batch_size, height, width))
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    boundary_predictions = torch.rand(batch_size, 1, height, width)
    boundary_targets = torch.randint(0, 2, (batch_size, height, width))
    
    # Initialize metrics
    metrics = IrisSegmentationMetrics(num_classes=num_classes)
    
    # Update with dummy data
    metrics.update(predictions, targets, boundary_predictions, boundary_targets)
    
    # Compute all metrics
    all_metrics = metrics.compute_all_metrics()
    
    print("Computed metrics:")
    for key, value in all_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nMetrics computation test completed successfully!")

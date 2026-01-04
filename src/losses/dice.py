"""
Dice Loss implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Multiclass Dice Loss for segmentation
    """
    
    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [B, C, H, W] logits
            targets: [B, H, W] class indices
        
        Returns:
            Dice loss value
        """
        # Convert logits to probabilities
        predictions = F.softmax(predictions, dim=1)
        
        # One-hot encode targets
        num_classes = predictions.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Compute Dice coefficient for each class
        dice_scores = []
        for c in range(num_classes):
            pred_c = predictions[:, c:c+1, :, :]
            target_c = targets_one_hot[:, c:c+1, :, :]
            
            intersection = (pred_c * target_c).sum(dim=(2, 3))
            union = pred_c.sum(dim=(2, 3)) + target_c.sum(dim=(2, 3))
            
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        dice_scores = torch.stack(dice_scores, dim=1)  # [B, C]
        
        # Average across classes and batch
        if self.reduction == 'mean':
            return 1 - dice_scores.mean()
        elif self.reduction == 'sum':
            return (1 - dice_scores).sum()
        else:
            return 1 - dice_scores


class GeneralizedDiceLoss(nn.Module):
    """
    Generalized Dice Loss with class weighting
    """
    
    def __init__(self, smooth: float = 1e-6, weight_type: str = 'inverse'):
        super().__init__()
        self.smooth = smooth
        self.weight_type = weight_type
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [B, C, H, W] logits
            targets: [B, H, W] class indices
        
        Returns:
            Generalized Dice loss value
        """
        predictions = F.softmax(predictions, dim=1)
        num_classes = predictions.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Calculate class weights
        if self.weight_type == 'inverse':
            # Inverse of class frequency
            class_counts = targets_one_hot.sum(dim=(0, 2, 3))
            weights = 1.0 / (class_counts + self.smooth)
        elif self.weight_type == 'sqrt':
            # Square root of inverse frequency
            class_counts = targets_one_hot.sum(dim=(0, 2, 3))
            weights = 1.0 / torch.sqrt(class_counts + self.smooth)
        else:
            weights = torch.ones(num_classes, device=predictions.device)
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Compute weighted Dice
        numerator = 0
        denominator = 0
        
        for c in range(num_classes):
            pred_c = predictions[:, c, :, :]
            target_c = targets_one_hot[:, c, :, :]
            
            intersection = (pred_c * target_c).sum()
            pred_sum = pred_c.sum()
            target_sum = target_c.sum()
            
            numerator += weights[c] * intersection
            denominator += weights[c] * (pred_sum + target_sum)
        
        dice = (2 * numerator + self.smooth) / (denominator + self.smooth)
        
        return 1 - dice

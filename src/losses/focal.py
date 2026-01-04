"""
Focal Loss implementations for handling class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [B, C, H, W] logits
            targets: [B, H, W] class indices
        
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss for extreme class imbalance
    """
    
    def __init__(
        self,
        alpha_pos: float = 0.25,
        alpha_neg: float = 0.75,
        gamma_pos: float = 2.0,
        gamma_neg: float = 4.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [B, C, H, W] logits
            targets: [B, H, W] class indices
        
        Returns:
            Asymmetric focal loss value
        """
        # Convert to probabilities
        probs = F.softmax(predictions, dim=1)
        
        # For binary case (background/pupil vs iris)
        if predictions.shape[1] == 2:
            # Get probabilities for each class
            p_neg = probs[:, 0, :, :]  # Background/pupil
            p_pos = probs[:, 1, :, :]  # Iris
            
            # Create masks for positive and negative samples
            pos_mask = (targets == 1).float()
            neg_mask = (targets == 0).float()
            
            # Compute focal weights
            focal_weight_pos = self.alpha_pos * (1 - p_pos) ** self.gamma_pos
            focal_weight_neg = self.alpha_neg * p_pos ** self.gamma_neg
            
            # Compute losses
            pos_loss = -focal_weight_pos * torch.log(p_pos + 1e-8) * pos_mask
            neg_loss = -focal_weight_neg * torch.log(p_neg + 1e-8) * neg_mask
            
            total_loss = pos_loss + neg_loss
        else:
            # Fallback to standard focal loss for multi-class
            ce_loss = F.cross_entropy(predictions, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            total_loss = self.alpha_pos * (1 - pt) ** self.gamma_pos * ce_loss
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice loss with adjustable precision/recall weighting
    """
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [B, C, H, W] logits
            targets: [B, H, W] class indices
        
        Returns:
            Tversky loss value
        """
        predictions = F.softmax(predictions, dim=1)
        num_classes = predictions.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        tversky_scores = []
        
        for c in range(num_classes):
            pred_c = predictions[:, c, :, :]
            target_c = targets_one_hot[:, c, :, :]
            
            # True positives, false positives, false negatives
            tp = (pred_c * target_c).sum()
            fp = (pred_c * (1 - target_c)).sum()
            fn = ((1 - pred_c) * target_c).sum()
            
            tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            tversky_scores.append(tversky)
        
        tversky_scores = torch.stack(tversky_scores)
        
        return 1 - tversky_scores.mean()


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss combining benefits of Focal and Tversky losses
    """
    
    def __init__(
        self, 
        alpha: float = 0.7, 
        beta: float = 0.3, 
        gamma: float = 1.3, 
        smooth: float = 1e-6
    ):
        super().__init__()
        self.tversky_loss = TverskyLoss(alpha, beta, smooth)
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [B, C, H, W] logits
            targets: [B, H, W] class indices
        
        Returns:
            Focal Tversky loss value
        """
        tversky = 1 - self.tversky_loss(predictions, targets)  # Get Tversky index
        focal_tversky = tversky ** self.gamma
        
        return 1 - focal_tversky

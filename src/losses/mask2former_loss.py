"""
Combined loss functions for iris segmentation
Implements CE + Dice + BoundaryIoU as recommended by Oracle
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import numpy as np


class DiceLoss(nn.Module):
    """
    Multiclass Dice Loss
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


class BoundaryIoULoss(nn.Module):
    """
    Boundary IoU Loss for sharp boundary prediction
    """
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, boundary_pred: torch.Tensor, boundary_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            boundary_pred: [B, 1, H, W] boundary logits
            boundary_target: [B, H, W] or [B, 1, H, W] boundary ground truth
        
        Returns:
            Boundary IoU loss
        """
        # Ensure boundary_target has correct shape
        if boundary_target.dim() == 3:
            boundary_target = boundary_target.unsqueeze(1)
        
        # Convert logits to probabilities
        boundary_pred = torch.sigmoid(boundary_pred)
        boundary_target = boundary_target.float()
        
        # Compute IoU
        intersection = (boundary_pred * boundary_target).sum(dim=(2, 3))
        union = boundary_pred.sum(dim=(2, 3)) + boundary_target.sum(dim=(2, 3)) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou.mean()


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


class CombinedIrisLoss(nn.Module):
    """
    Combined loss function for iris segmentation
    Implements: 0.5 * CE + 0.5 * Dice + 0.25 * BoundaryIoU
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        boundary_weight: float = 0.25,
        aux_weight: float = 0.2,
        use_focal: bool = False,
        focal_alpha: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.aux_weight = aux_weight
        self.use_focal = use_focal
        
        # Loss components
        if use_focal:
            self.ce_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryIoULoss()
        
        # For auxiliary loss (deep supervision)
        if use_focal:
            self.aux_ce_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.aux_ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Dictionary containing:
                - logits: [B, C, H, W] main segmentation logits
                - boundary_logits: [B, 1, H, W] boundary logits (optional)
                - aux_logits: [B, C, H, W] auxiliary logits (optional)
            targets: Dictionary containing:
                - labels: [B, H, W] segmentation targets
                - boundary: [B, H, W] boundary targets (optional)
        
        Returns:
            Dictionary with loss components and total loss
        """
        labels = targets['labels']
        boundary = targets.get('boundary', None)
        
        losses = {}
        
        # Main segmentation loss
        ce_loss = self.ce_loss(outputs['logits'], labels)
        dice_loss = self.dice_loss(outputs['logits'], labels)
        
        losses['ce_loss'] = ce_loss
        losses['dice_loss'] = dice_loss
        
        main_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        # Boundary loss
        if 'boundary_logits' in outputs and boundary is not None:
            boundary_loss = self.boundary_loss(outputs['boundary_logits'], boundary)
            losses['boundary_loss'] = boundary_loss
            main_loss += self.boundary_weight * boundary_loss
        
        # Auxiliary loss (deep supervision)
        if 'aux_logits' in outputs:
            aux_loss = self.aux_ce_loss(outputs['aux_logits'], labels)
            losses['aux_loss'] = aux_loss
            main_loss += self.aux_weight * aux_loss
        
        losses['total_loss'] = main_loss
        
        return losses


class AdaptiveWeightedLoss(nn.Module):
    """
    Adaptive loss that adjusts weights based on training progress
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        warmup_epochs: int = 10,
        adapt_boundary: bool = True
    ):
        super().__init__()
        self.base_loss = base_loss
        self.warmup_epochs = warmup_epochs
        self.adapt_boundary = adapt_boundary
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """Set current epoch for adaptive weighting"""
        self.current_epoch = epoch
        
        if self.adapt_boundary and hasattr(self.base_loss, 'boundary_weight'):
            # Gradually increase boundary weight after warmup
            if epoch < self.warmup_epochs:
                self.base_loss.boundary_weight = 0.1
            else:
                progress = min(1.0, (epoch - self.warmup_epochs) / self.warmup_epochs)
                self.base_loss.boundary_weight = 0.1 + 0.15 * progress  # 0.1 -> 0.25
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.base_loss(outputs, targets)


def create_loss_function(
    num_classes: int = 2,
    class_distribution: Optional[torch.Tensor] = None,
    loss_type: str = "combined",  # "combined", "focal", "adaptive"
    device: torch.device = torch.device('cpu'),
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions
    
    Args:
        num_classes: Number of segmentation classes
        class_distribution: Class pixel distribution for weight calculation
        loss_type: Type of loss function
        device: Device to place tensors on
        **kwargs: Additional loss function arguments
    
    Returns:
        Loss function instance
    """
    
    # Calculate class weights if distribution provided
    class_weights = None
    if class_distribution is not None:
        total_pixels = class_distribution.sum()
        class_weights = total_pixels / (num_classes * class_distribution)
        class_weights = class_weights.to(device)
        print(f"Calculated class weights: {class_weights}")
    
    if loss_type == "combined":
        loss_fn = CombinedIrisLoss(
            class_weights=class_weights,
            **kwargs
        )
    elif loss_type == "focal":
        loss_fn = CombinedIrisLoss(
            class_weights=None,  # Focal loss handles imbalance differently
            use_focal=True,
            focal_alpha=class_weights,
            **kwargs
        )
    elif loss_type == "adaptive":
        base_loss = CombinedIrisLoss(
            class_weights=class_weights,
            **kwargs
        )
        loss_fn = AdaptiveWeightedLoss(
            base_loss=base_loss,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    return loss_fn


def test_loss_functions():
    """Test loss function implementations"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    num_classes = 2
    height, width = 128, 128
    
    # Create dummy data
    logits = torch.randn(batch_size, num_classes, height, width, device=device)
    boundary_logits = torch.randn(batch_size, 1, height, width, device=device)
    aux_logits = torch.randn(batch_size, num_classes, height, width, device=device)
    
    labels = torch.randint(0, num_classes, (batch_size, height, width), device=device)
    boundary = torch.randint(0, 2, (batch_size, height, width), device=device)
    
    outputs = {
        'logits': logits,
        'boundary_logits': boundary_logits,
        'aux_logits': aux_logits
    }
    
    targets = {
        'labels': labels,
        'boundary': boundary
    }
    
    # Test individual losses
    print("Testing individual loss components...")
    
    dice_loss = DiceLoss()
    dice_val = dice_loss(logits, labels)
    print(f"Dice Loss: {dice_val.item():.4f}")
    
    boundary_loss = BoundaryIoULoss()
    boundary_val = boundary_loss(boundary_logits, boundary)
    print(f"Boundary IoU Loss: {boundary_val.item():.4f}")
    
    focal_loss = FocalLoss()
    focal_val = focal_loss(logits, labels)
    print(f"Focal Loss: {focal_val.item():.4f}")
    
    # Test combined loss
    print("\nTesting combined loss...")
    class_dist = torch.tensor([0.93, 0.07])  # Background/pupil: 93%, Iris: 7%
    combined_loss = create_loss_function(
        num_classes=2,
        class_distribution=class_dist,
        loss_type="combined",
        device=device
    )
    
    loss_dict = combined_loss(outputs, targets)
    print("Combined loss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    
    print("\nLoss function tests completed successfully!")


if __name__ == "__main__":
    test_loss_functions()

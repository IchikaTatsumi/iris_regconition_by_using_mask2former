"""
Combined loss functions implementing Oracle's recommendations
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from .dice import DiceLoss, GeneralizedDiceLoss
from .boundary import BoundaryIoULoss, BoundaryDiceLoss
from .focal import FocalLoss, FocalTverskyLoss


class CombinedIrisLoss(nn.Module):
    """
    Combined loss function for iris segmentation
    Implements: 0.5 * CE + 0.5 * Dice + 0.25 * BoundaryIoU (Oracle recommendation)
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
        focal_gamma: float = 2.0,
        dice_type: str = 'standard'  # 'standard' or 'generalized'
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
        
        if dice_type == 'generalized':
            self.dice_loss = GeneralizedDiceLoss()
        else:
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


class UnifiedIrisLoss(nn.Module):
    """
    Unified loss combining multiple loss types with automatic weighting
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        auto_weight: bool = False
    ):
        super().__init__()
        
        # Default loss weights (Oracle's recommendation)
        if loss_weights is None:
            loss_weights = {
                'ce': 0.5,
                'dice': 0.5,
                'boundary': 0.25,
                'focal_tversky': 0.0  # Optional
            }
        
        self.loss_weights = loss_weights
        self.auto_weight = auto_weight
        
        # Initialize loss functions
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryIoULoss()
        self.focal_tversky_loss = FocalTverskyLoss()
        
        # For automatic weighting
        if auto_weight:
            self.loss_weights_learnable = nn.Parameter(
                torch.tensor(list(loss_weights.values()))
            )
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute unified loss with multiple components
        """
        labels = targets['labels']
        boundary = targets.get('boundary', None)
        
        losses = {}
        
        # Compute individual losses
        ce_loss = self.ce_loss(outputs['logits'], labels)
        dice_loss = self.dice_loss(outputs['logits'], labels)
        
        losses['ce_loss'] = ce_loss
        losses['dice_loss'] = dice_loss
        
        # Combine main losses
        if self.auto_weight:
            weights = F.softmax(self.loss_weights_learnable, dim=0)
            total_loss = weights[0] * ce_loss + weights[1] * dice_loss
        else:
            total_loss = self.loss_weights['ce'] * ce_loss + self.loss_weights['dice'] * dice_loss
        
        # Add boundary loss if available
        if 'boundary_logits' in outputs and boundary is not None:
            boundary_loss = self.boundary_loss(outputs['boundary_logits'], boundary)
            losses['boundary_loss'] = boundary_loss
            
            boundary_weight = weights[2] if self.auto_weight else self.loss_weights['boundary']
            total_loss += boundary_weight * boundary_loss
        
        # Add focal tversky if enabled
        if self.loss_weights.get('focal_tversky', 0) > 0:
            ft_loss = self.focal_tversky_loss(outputs['logits'], labels)
            losses['focal_tversky_loss'] = ft_loss
            
            ft_weight = weights[3] if self.auto_weight else self.loss_weights['focal_tversky']
            total_loss += ft_weight * ft_loss
        
        losses['total_loss'] = total_loss
        
        # Log learnable weights if using auto weighting
        if self.auto_weight:
            weights = F.softmax(self.loss_weights_learnable, dim=0)
            losses['weight_ce'] = weights[0]
            losses['weight_dice'] = weights[1]
            if len(weights) > 2:
                losses['weight_boundary'] = weights[2]
        
        return losses


def create_loss_function(
    num_classes: int = 2,
    class_distribution: Optional[torch.Tensor] = None,
    class_weights: Optional[torch.Tensor] = None,  # Pre-calculated weights (takes precedence)
    loss_type: str = "combined",  # "combined", "focal", "adaptive", "unified"
    device: torch.device = torch.device('cpu'),
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions
    
    Args:
        num_classes: Number of segmentation classes
        class_distribution: Class pixel distribution for weight calculation
        class_weights: Pre-calculated class weights (takes precedence over distribution)
        loss_type: Type of loss function
        device: Device to place tensors on
        **kwargs: Additional loss function arguments
    
    Returns:
        Loss function instance
    """
    
    # Use pre-calculated weights if provided, otherwise calculate from distribution
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print(f"✅ Using pre-calculated class weights: {class_weights}")
    elif class_distribution is not None:
        total_pixels = class_distribution.sum()
        class_weights = total_pixels / (num_classes * class_distribution)
        class_weights = class_weights.to(device)
        print(f"📊 Calculated class weights from distribution: {class_weights}")
    else:
        print("⚠️  No class weights or distribution provided - using unweighted loss")
    
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
    elif loss_type == "unified":
        loss_fn = UnifiedIrisLoss(
            class_weights=class_weights,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    return loss_fn

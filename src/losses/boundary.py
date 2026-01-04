"""
Boundary-aware loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


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


class BoundaryDiceLoss(nn.Module):
    """
    Dice loss specifically for boundary prediction
    """
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, boundary_pred: torch.Tensor, boundary_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            boundary_pred: [B, 1, H, W] boundary logits
            boundary_target: [B, H, W] boundary ground truth
        
        Returns:
            Boundary Dice loss
        """
        # Ensure shapes match
        if boundary_target.dim() == 3:
            boundary_target = boundary_target.unsqueeze(1)
        
        boundary_pred = torch.sigmoid(boundary_pred)
        boundary_target = boundary_target.float()
        
        # Flatten for computation
        pred_flat = boundary_pred.view(boundary_pred.shape[0], -1)
        target_flat = boundary_target.view(boundary_target.shape[0], -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()


class ActiveContourLoss(nn.Module):
    """
    Active Contour Loss for smooth boundary prediction
    """
    
    def __init__(self, mu: float = 1.0):
        super().__init__()
        self.mu = mu
    
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [B, C, H, W] segmentation probabilities
        
        Returns:
            Active contour loss encouraging smooth boundaries
        """
        # Take the foreground class (iris)
        if predictions.shape[1] > 1:
            predictions = F.softmax(predictions, dim=1)[:, 1:2, :, :]  # [B, 1, H, W]
        
        # Compute gradient of predictions
        grad_x = torch.abs(predictions[:, :, :, :-1] - predictions[:, :, :, 1:])
        grad_y = torch.abs(predictions[:, :, :-1, :] - predictions[:, :, 1:, :])
        
        # Length term (encourages smooth boundaries)
        length = grad_x.mean() + grad_y.mean()
        
        return self.mu * length


def create_boundary_mask(mask: torch.Tensor, dilation_size: int = 3) -> torch.Tensor:
    """
    Create boundary mask for boundary-aware loss
    
    Args:
        mask: Binary mask [H, W] or [B, H, W] with values {0, 1}
        dilation_size: Size of morphological kernel
    
    Returns:
        Boundary mask with boundary pixels = 1
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)  # Add batch dimension
    
    boundary_masks = []
    
    for i in range(mask.shape[0]):
        mask_np = mask[i].cpu().numpy().astype(np.uint8)
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (dilation_size, dilation_size)
        )
        
        # Create boundary by dilation - erosion
        dilated = cv2.dilate(mask_np, kernel, iterations=1)
        eroded = cv2.erode(mask_np, kernel, iterations=1)
        boundary = dilated ^ eroded  # XOR to get boundary
        
        boundary_masks.append(torch.from_numpy(boundary).to(mask.device))
    
    return torch.stack(boundary_masks, dim=0)


class EdgeAwareLoss(nn.Module):
    """
    Edge-aware loss that emphasizes boundary regions
    """
    
    def __init__(self, edge_weight: float = 2.0, smooth: float = 1e-6):
        super().__init__()
        self.edge_weight = edge_weight
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [B, C, H, W] logits
            targets: [B, H, W] class indices
        
        Returns:
            Edge-aware cross entropy loss
        """
        # Create edge map from targets
        edge_map = create_boundary_mask(targets).float()
        
        # Standard cross entropy
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # Weight by edge map
        weighted_loss = ce_loss * (1 + self.edge_weight * edge_map)
        
        return weighted_loss.mean()

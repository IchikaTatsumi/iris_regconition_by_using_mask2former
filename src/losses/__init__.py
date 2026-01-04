"""
Loss functions for iris segmentation
FIXED: Import from correct module
"""

from .dice import DiceLoss
from .boundary import BoundaryIoULoss
from .focal import FocalLoss
from .mask2former_loss import (
    CombinedIrisLoss,
    AdaptiveWeightedLoss,
    create_mask2former_loss  # FIXED: Renamed for clarity
)

__all__ = [
    'DiceLoss',
    'BoundaryIoULoss',
    'FocalLoss',
    'CombinedIrisLoss',
    'AdaptiveWeightedLoss',
    'create_mask2former_loss'
]

"""
Model components for iris segmentation
FIXED: Correct imports from mask2former module
"""

from .mask2former import EnhancedMask2Former, create_model
from .heads import BoundaryRefinementHead

__all__ = [
    'EnhancedMask2Former',
    'BoundaryRefinementHead',
    'create_model'
]
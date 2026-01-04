"""
FIXED: Inference module with correct class names
"""

from .mask2former_inference import (
    Mask2FormerInference,  # FIXED: Correct class name
    load_inference_model,
    quick_inference
)

__all__ = [
    'Mask2FormerInference',  # FIXED: Not IrisSegmentationInference
    'load_inference_model',
    'quick_inference'
]
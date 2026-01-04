"""
Inference module for iris segmentation using trained SegFormer models
"""

from .mask2former_inference import (
    IrisSegmentationInference,
    load_inference_model,
    quick_inference
)

__all__ = [
    'IrisSegmentationInference',
    'load_inference_model', 
    'quick_inference'
]

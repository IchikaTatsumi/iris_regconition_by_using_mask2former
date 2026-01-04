# Iris Segmentation Inference Package

This package provides inference capabilities for trained SegFormer iris segmentation models.

## Package Structure

```
src/inference/
├── __init__.py          # Package initialization and exports
├── inference.py         # Main inference implementation
└── README.md           # This file
```

## Exported Classes and Functions

- **`IrisSegmentationInference`** - Main inference class
- **`load_inference_model`** - Convenience function to load inference model
- **`quick_inference`** - Quick inference function for single images

## Usage

```python
# Import from the inference package
from src.inference import IrisSegmentationInference

# Or when src is in Python path
from inference import IrisSegmentationInference

# Load model and run inference
model = IrisSegmentationInference('path/to/checkpoint.pt')
results = model.predict('path/to/image.jpg')
```

## Features

- ✅ Load trained SegFormer models from checkpoints
- ✅ Single image and batch processing
- ✅ Segmentation and boundary prediction support
- ✅ Automatic GPU/CPU detection
- ✅ Comprehensive result saving and visualization
- ✅ Confidence scoring and thresholding

## Integration

This package is designed to work seamlessly with the trained models from the iris segmentation training pipeline. It automatically handles:

- Model architecture reconstruction
- Checkpoint loading with proper state dict mapping
- Image preprocessing with the same transforms used in training
- Result postprocessing and visualization

See the main `INFERENCE_GUIDE.md` for complete usage instructions.

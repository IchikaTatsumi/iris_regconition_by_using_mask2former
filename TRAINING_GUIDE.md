# SegFormer Iris Segmentation Training Guide

## Overview

Complete implementation of SegFormer training for iris segmentation on UBIRIS V2 dataset, following Oracle's comprehensive analysis and recommendations.

## 🎯 Expected Performance
- **Target mIoU**: ≥0.90
- **Target Dice**: ≥0.93 
- **Inference Speed**: ~45 FPS on RTX 3090
- **Memory**: Fits on 8-12GB GPU

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Calculate Class Weights (IMPORTANT for good results)
```bash
# Run this BEFORE training to fix class imbalance
python class_weights_util.py
```

### 3. Quick Training (Recommended)
```bash
python train.py --epochs 160 --batch-size 8 --wandb
```

### 4. Advanced Training with Config
```bash
python train.py --config configs/segformer_iris_config.json --wandb
```

### 5. Resume Training (Automatic)
Training automatically resumes from the last checkpoint if available:
```bash
# Will automatically find and resume from outputs/checkpoints/last.pt
python train.py --epochs 160 --batch-size 8 --wandb
```

## 🚨 Class Imbalance Fix (CRITICAL)

### The Problem
Iris segmentation datasets have severe class imbalance:
- **Background/Pupil**: 94.7% of pixels
- **Iris**: 5.3% of pixels

This causes models to predict "all background" for better loss, resulting in poor iris segmentation.

### The Solution
**Automatic Class Weighting** - We've integrated an automatic class balancing system:

#### 1. Calculate Optimal Weights
```bash
python class_weights_util.py
```

This analyzes your dataset and calculates optimal class weights:
- **Background weight**: 0.53 (reduce importance)
- **Iris weight**: 9.43 (greatly increase importance)

#### 2. Training Pipeline Integration
The training system automatically loads these weights:

```python
# Training automatically uses weights from class_weights.pt
weights = torch.tensor([0.5280, 9.4296])  # [background, iris]
criterion = nn.CrossEntropyLoss(weight=weights)
```

#### 3. Supported Loss Functions
- **Weighted Cross-Entropy**: `nn.CrossEntropyLoss(weight=weights)`
- **Focal Loss**: Handles hard examples + imbalance
- **Dice Loss**: Excellent for segmentation
- **Combined Loss**: CE + Dice + Boundary (recommended)

#### 4. Expected Improvement
- **Before**: Confusion matrix shows ~99% background predictions
- **After**: Balanced predictions with proper iris segmentation

**⚠️ IMPORTANT**: Run `python class_weights_util.py` before training for best results!

## 📁 Project Structure

```
iris_recognition/
├── configs/
│   └── segformer_iris_config.json    # Training configuration
├── src/
│   ├── models.py                     # Enhanced SegFormer with boundary head
│   ├── losses.py                     # Combined CE+Dice+BoundaryIoU loss
│   ├── metrics.py                    # Comprehensive evaluation metrics
│   ├── transforms.py                 # Advanced augmentation pipeline
│   ├── trainer.py                    # Main training orchestrator
│   └── data/
│       ├── dataset.py               # Enhanced dataset with subject splits
│       └── dataloader.py            # DataLoader utilities
├── dataset/                         # UBIRIS V2 data
│   ├── images/
│   └── masks/
├── outputs/                        # Auto-created training outputs
│   └── checkpoints/
│       ├── last.pt                 # Latest checkpoint (auto-resume)
│       ├── best.pt                 # Best model checkpoint
│       └── epoch_*.pth            # Periodic checkpoints
├── train.py                        # Training entry point
├── requirements.txt                # Python dependencies
└── TRAINING_GUIDE.md              # This file
```

## 🔧 Key Enhancements (Oracle Recommendations)

### 1. **Aspect Ratio Preservation**
```python
# OLD: Distorting resize (300x400 → 512x512)
transforms.Resize((512, 512))

# NEW: Preserve aspect ratio with padding
AspectRatioPreservingResize(target_size=512)
```

### 2. **Advanced Augmentation**
- ✅ Horizontal/vertical flips
- ✅ Rotation (±10°) 
- ✅ Scale variation (0.9-1.1)
- ✅ Color jittering
- ✅ Gaussian blur (simulates focus issues)
- ✅ Safe augmentations that preserve iris structure

### 3. **Subject-Aware Data Splitting**
```python
# Prevents data leakage by splitting on subject ID
# Train: 80% subjects, Val: 10% subjects, Test: 10% subjects
UbirisDataset(use_subject_split=True)
```

### 4. **Enhanced SegFormer Architecture**
```python
# SegFormer-B1 + Boundary Refinement Head
model = EnhancedSegFormer(
    model_name="nvidia/segformer-b1-finetuned-ade-512-512",
    add_boundary_head=True,
    freeze_encoder=True,
    freeze_epochs=10
)
```

### 5. **Multi-Component Loss Function**
```python
# Combined loss addressing class imbalance and boundary sharpness
Loss = 0.5 * CE(weighted) + 0.5 * Dice + 0.25 * BoundaryIoU
```

### 6. **Optimal Training Configuration**
- **Optimizer**: AdamW with weight decay 0.01
- **Learning Rate**: 3e-5 with polynomial decay
- **Batch Size**: 8 (auto-adjusted based on GPU memory)
- **Epochs**: 160 with early stopping (patience=15)
- **Mixed Precision**: Enabled for memory efficiency
- **Auto-Resume**: Automatically resumes from last.pt if available

### 7. **Checkpoint Management**
- **Auto-Creation**: Output directories created automatically
- **last.pt**: Saved after every epoch for resume capability
- **best.pt**: Best performing model saved automatically
- **epoch_*.pth**: Periodic checkpoints every 50 epochs

## 📊 Training Configuration Details

### Model Configuration
```json
{
  "model_name": "nvidia/segformer-b1-finetuned-ade-512-512",
  "model_type": "enhanced",
  "num_labels": 2,
  "add_boundary_head": true,
  "freeze_encoder": true,
  "freeze_epochs": 10
}
```

### Loss Configuration
```json
{
  "loss_type": "combined",
  "ce_weight": 0.5,
  "dice_weight": 0.5,
  "boundary_weight": 0.25,
  "use_focal": false
}
```

### Data Configuration
```json
{
  "batch_size": 8,
  "use_subject_split": true,
  "preserve_aspect": true,
  "image_size": 512,
  "num_workers": 4
}
```

## 🔍 Monitoring and Evaluation

### Key Metrics Tracked
1. **Primary**: Mean IoU (mIoU)
2. **Secondary**: Dice coefficient
3. **Boundary**: Boundary F1 score
4. **Speed**: Inference FPS

### Weights & Biases Integration
```python
# Automatic logging of:
# - Loss components (CE, Dice, Boundary)
# - Metrics (IoU, Dice, F1)
# - Learning rate schedules
# - Model parameters and gradients
```

### Early Stopping & Resume
- **Monitor**: Validation mIoU
- **Patience**: 15 epochs
- **Direction**: Maximize
- **Auto-Resume**: Training resumes from last.pt automatically if training is interrupted

## 🎮 Usage Examples

### Basic Training
```python
import sys
sys.path.append('src')
from trainer import IrisSegmentationTrainer, create_dataloaders

# Load config
with open('configs/segformer_iris_config.json') as f:
    config = json.load(f)

# Create dataloaders
dataloaders = create_dataloaders(config)

# Create trainer
trainer = IrisSegmentationTrainer(config, use_wandb=True)

# Train
trainer.train(dataloaders['train'], dataloaders['val'])
```

### Custom Configuration
```python
# Modify config for your needs
config['training']['num_epochs'] = 200
config['data']['batch_size'] = 4
config['model']['freeze_epochs'] = 0  # No encoder freezing

trainer = IrisSegmentationTrainer(config)
trainer.train(train_loader, val_loader)
```

### Resume from Checkpoint
```python
# Manual checkpoint loading (if needed)
trainer = IrisSegmentationTrainer(
    config, 
    resume_from='outputs/checkpoints/last.pt'
)
trainer.train(train_loader, val_loader)

# Note: Auto-resume happens by default when using train.py
```

## 🔧 Troubleshooting

### Common Issues

#### 1. **GPU Memory Error**
```bash
# Reduce batch size
python train.py --batch-size 4

# Or use gradient accumulation
# (modify config: add "gradient_accumulation_steps": 2)
```

#### 2. **Import Errors**
```bash
# Ensure proper Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python train.py
```

#### 3. **Missing Dependencies**
```bash
pip install -r requirements.txt
```

#### 4. **Poor Confusion Matrix / All Background Predictions**
```bash
# CRITICAL FIX: Calculate class weights first
python class_weights_util.py

# Then verify data pipeline
python debug_data_pipeline.py

# Check that class_weights.pt exists and contains reasonable weights
```

**Root Causes:**
- Severe class imbalance (94.7% background vs 5.3% iris)
- Mask preprocessing issues (anti-aliasing)
- Boundary transform misalignment
- Missing class weights in loss function

**Solution Applied:**
✅ Threshold-based mask conversion (`mask > 127`)
✅ Synchronized boundary transforms
✅ Automatic class weight loading
✅ Improved subject-aware data splitting

#### 5. **Dataset Not Found**
```bash
# Ensure dataset structure
ls -la dataset/
# Should show: images/ and masks/ directories
```

#### 5. **Training Interrupted**
```bash
# Training automatically resumes from last.pt
python train.py --epochs 160 --batch-size 8 --wandb
# ✅ Will automatically find and resume from outputs/checkpoints/last.pt
```

#### 6. **Missing Output Directory**
```bash
# Output directories are created automatically
# No manual creation needed - the system handles this
```

#### 7. **Slow Training**
```bash
# Increase number of workers
python train.py --config configs/segformer_iris_config.json
# Then modify config: "num_workers": 8
```

### Performance Optimization

#### For Low-End GPUs (≤8GB)
```json
{
  "data": {"batch_size": 4},
  "model": {"model_name": "nvidia/segformer-b0-finetuned-ade-512-512"}
}
```

#### For High-End GPUs (≥16GB)
```json
{
  "data": {"batch_size": 12},
  "model": {"model_name": "nvidia/segformer-b2-finetuned-ade-512-512"}
}
```

## 📈 Expected Training Timeline

### SegFormer-B1 on RTX 3090
- **Setup Time**: ~2 minutes
- **Epoch Time**: ~15 minutes (1800 samples, batch=8)
- **Total Training**: ~40 hours (160 epochs)
- **Early Stopping**: Typically ~100-120 epochs

### Memory Usage
- **SegFormer-B1**: ~9GB VRAM (batch=8)
- **SegFormer-B0**: ~6GB VRAM (batch=8)
- **SegFormer-B2**: ~13GB VRAM (batch=8)

## 🎯 Results Interpretation

### Good Results Indicators
- **Validation mIoU**: >0.85
- **Validation Dice**: >0.90
- **Boundary F1**: >0.80
- **Loss Convergence**: Steady decrease without oscillation

### Poor Results Indicators
- **mIoU plateau**: <0.75 after 50 epochs
- **High variance**: Large differences between batches
- **Overfitting**: Train mIoU >> Val mIoU

## 🔄 Advanced Features

### 5-Fold Cross Validation
```python
# Future enhancement
from trainer import CrossValidationTrainer
cv_trainer = CrossValidationTrainer(config, n_folds=5)
cv_results = cv_trainer.train()
```

### Model Ensembling
```python
# Load multiple trained models
models = [load_model(f'fold_{i}_best.pth') for i in range(5)]
ensemble_predictions = ensemble_predict(models, test_loader)
```

### Export for Deployment
```python
# Export to ONNX
torch.onnx.export(model, dummy_input, 'iris_segformer.onnx')

# TensorRT optimization
import tensorrt as trt
# ... conversion code
```

## 🎓 Tips for Best Results

1. **Start Small**: Use batch_size=4 and shorter epochs first
2. **Monitor Closely**: Watch for overfitting in first 20 epochs
3. **Adjust Learning Rate**: If loss plateaus, try 1e-5 or 5e-5
4. **Use Early Stopping**: Don't overtrain
5. **Auto-Resume**: Training resumes automatically - no need to worry about interruptions
6. **Visualize Results**: Check sample predictions regularly
7. **Checkpoint Management**: System automatically saves last.pt and best.pt

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the Oracle's analysis in previous conversation
3. Check training logs and metrics
4. Verify dataset preprocessing with `test_enhanced_dataset.py`

---

**Implementation Date**: September 10, 2025  
**Based on**: Oracle's comprehensive SegFormer analysis  
**Status**: Ready for production training 🚀

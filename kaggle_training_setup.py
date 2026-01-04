#!/usr/bin/env python3
"""
Setup script for Kaggle GPU training
Creates optimized config and prepares dataset
"""

import json
import shutil
from pathlib import Path

def create_kaggle_config():
    """Create optimized config for Kaggle GPU (P100/T4)"""
    
    print("🔧 Creating Kaggle-optimized configuration...")
    
    # Kaggle GPU specs:
    # - P100: 16GB VRAM
    # - T4: 16GB VRAM
    # - Time limit: 12 hours per session
    
    config = {
        "experiment_name": "mask2former_iris_kaggle",
        "description": "Mask2Former for iris segmentation - Kaggle optimized",
        
        "model": {
            "architecture": "mask2former",
            "model_name": "facebook/mask2former-swin-small-coco-panoptic",
            "model_type": "enhanced",
            "num_labels": 2,
            "num_queries": 50,
            "hidden_dim": 256,
            "add_boundary_head": True,
            "freeze_backbone": True,
            "freeze_epochs": 5,  # Less than local
            "use_checkpoint": False
        },
        
        "data": {
            "dataset_dir": "/kaggle/input/your-dataset-name/dataset",  # UPDATE THIS
            "dataset_root": "/kaggle/input/your-dataset-name/dataset",
            "images_dir": "/kaggle/input/your-dataset-name/dataset/images",
            "masks_dir": "/kaggle/input/your-dataset-name/dataset/masks",
            "image_size": 512,
            "train_val_split": 0.85,
            "subject_aware_split": True,
            "use_subject_split": True,
            "preserve_aspect_ratio": True,
            "preserve_aspect": True,
            "num_workers": 2,  # Kaggle CPU cores
            "pin_memory": True,
            "batch_size": 8  # P100/T4 can handle this
        },
        
        "training": {
            "num_epochs": 100,  # Fit in 12 hours
            "epochs": 100,
            "batch_size": 8,
            "accumulation_steps": 1,  # No need with larger batch
            "learning_rate": 5e-5,
            "base_lr": 5e-5,
            "weight_decay": 0.01,
            "warmup_epochs": 3,  # Faster warmup
            "min_lr": 1e-7,
            "patience": 15,
            "save_frequency": 20,  # Save disk space
            "eval_freq": 1,
            "evaluate_every": 1,
            "log_freq": 50,  # Less frequent logging
            "gradient_clip": 1.0,
            "mixed_precision": True,
            "steps_per_epoch": 1000,
            
            "optimizer": {
                "type": "adamw",
                "base_lr": 5e-5,
                "learning_rate": 5e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            },
            
            "scheduler": {
                "type": "cosine",
                "warmup_steps": 300,  # Faster warmup
                "min_lr_ratio": 0.01,
                "min_lr": 1e-7,
                "power": 0.9
            },
            
            "early_stopping": {
                "enabled": True,
                "patience": 15,
                "min_delta": 0.001,
                "metric": "mean_iou",
                "mode": "max"
            },
            
            "gradient_clipping": {
                "enabled": True,
                "max_norm": 1.0
            }
        },
        
        "optimizer": {
            "type": "adamw",
            "base_lr": 5e-5,
            "learning_rate": 5e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        },
        
        "scheduler": {
            "type": "cosine",
            "warmup_steps": 300,
            "min_lr_ratio": 0.01,
            "min_lr": 1e-7,
            "power": 0.9
        },
        
        "loss": {
            "loss_type": "combined",
            "ce_weight": 0.5,
            "dice_weight": 0.5,
            "boundary_weight": 0.25,
            "mask2former_weight": 1.0,
            "aux_weight": 0.2,
            "use_focal": False,
            "focal_alpha": None,
            "focal_gamma": 2.0,
            "class_weights": [1.0, 15.0]
        },
        
        "augmentation": {
            "horizontal_flip": 0.5,
            "vertical_flip": 0.0,
            "rotation_limit": 10,
            "scale_limit": 0.1,
            "brightness_contrast": {
                "brightness_limit": 0.25,
                "contrast_limit": 0.25,
                "p": 0.4
            },
            "hue_saturation": {
                "hue_shift_limit": 10,
                "sat_shift_limit": 20,
                "val_shift_limit": 20,
                "p": 0.3
            },
            "clahe": {
                "clip_limit": 2.0,
                "tile_grid_size": [8, 8],
                "p": 0.2
            },
            "blur": {
                "blur_limit": 3,
                "p": 0.2
            },
            "noise": {
                "var_limit": 20.0,
                "p": 0.1
            }
        },
        
        "evaluation": {
            "metrics": [
                "pixel_accuracy",
                "mean_iou",
                "mean_dice",
                "class_iou",
                "class_dice",
                "precision",
                "recall",
                "f1_score",
                "boundary_f1"
            ],
            "evaluate_every": 1,
            "save_predictions": True,
            "save_best_model": True,
            "save_last_model": True
        },
        
        "checkpointing": {
            "save_dir": "/kaggle/working/outputs/mask2former_iris",
            "save_frequency": 20,
            "keep_best_n": 2,  # Save space
            "keep_last_n": 1,
            "monitor_metric": "mean_iou",
            "monitor_mode": "max"
        },
        
        "logging": {
            "use_wandb": False,  # Set to True if you have WandB API key
            "wandb_project": "iris-segmentation-mask2former",
            "wandb_entity": None,
            "log_frequency": 50,
            "log_images_frequency": 500,
            "log_gradients": False,
            "save_predictions_samples": 5
        },
        
        "visualization": {
            "enabled": True,
            "save_frequency": 20,
            "num_samples": 4,
            "save_dir": "/kaggle/working/outputs/mask2former_iris/visualizations"
        },
        
        "inference": {
            "checkpoint_path": "/kaggle/working/outputs/mask2former_iris/checkpoints/best.pt",
            "batch_size": 8,
            "device": "cuda",
            "use_amp": True,
            "confidence_threshold": 0.5
        },
        
        "hardware": {
            "device": "cuda",
            "gpu_ids": [0],
            "distributed": False,
            "world_size": 1,
            "find_unused_parameters": False
        },
        
        "reproducibility": {
            "seed": 42,
            "deterministic": False,
            "benchmark": True
        },
        
        "num_classes": 2,
        "class_names": ["background/pupil", "iris"],
        "class_distribution": [0.95, 0.05],
        "project_name": "iris-segmentation-mask2former-kaggle",
        "run_name": "mask2former-iris-kaggle",
        "tags": ["mask2former", "iris", "segmentation", "kaggle"],
        "output_dir": "/kaggle/working/outputs/mask2former_iris",
        
        "targets": {
            "mean_iou": 0.90,
            "mean_dice": 0.93,
            "iris_iou": 0.90,
            "boundary_f1": 0.80,
            "inference_fps": 30
        },
        
        "notes": {
            "kaggle_optimized": True,
            "gpu": "P100 or T4 (16GB)",
            "time_limit": "12 hours",
            "expected_training_time": "8-10 hours for 100 epochs",
            "batch_size": 8,
            "recommendations": [
                "Upload dataset as Kaggle dataset",
                "Update dataset paths in config",
                "Enable GPU in notebook settings",
                "Save checkpoints to /kaggle/working/",
                "Download best checkpoint before session ends",
                "Consider enabling WandB for monitoring"
            ]
        }
    }
    
    # Save config
    config_path = Path('configs/mask2former_config_kaggle.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Kaggle config saved: {config_path}")
    return config

def create_kaggle_notebook():
    """Create Kaggle notebook template"""
    
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Mask2Former Iris Segmentation - Kaggle Training\n",
                    "\n",
                    "## Setup\n",
                    "**Important**: Enable GPU in notebook settings (P100 or T4)\n"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Check GPU\n",
                    "!nvidia-smi\n",
                    "\n",
                    "import torch\n",
                    "print(f\"PyTorch version: {torch.__version__}\")\n",
                    "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
                    "if torch.cuda.is_available():\n",
                    "    print(f\"GPU: {torch.cuda.get_device_name()}\")\n",
                    "    print(f\"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB\")"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Install dependencies\n",
                    "!pip install -q transformers albumentations opencv-python\n",
                    "# Optional: WandB for logging\n",
                    "# !pip install -q wandb"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Clone your repository (if using GitHub)\n",
                    "# !git clone https://github.com/your-username/your-repo.git\n",
                    "# %cd your-repo\n",
                    "\n",
                    "# Or upload files directly to Kaggle dataset"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Verify dataset\n",
                    "import os\n",
                    "from pathlib import Path\n",
                    "\n",
                    "# UPDATE THIS PATH\n",
                    "dataset_path = Path('/kaggle/input/your-dataset-name/dataset')\n",
                    "\n",
                    "print(f\"Dataset exists: {dataset_path.exists()}\")\n",
                    "if dataset_path.exists():\n",
                    "    images = list((dataset_path / 'images').glob('*'))\n",
                    "    masks = list((dataset_path / 'masks').glob('*'))\n",
                    "    print(f\"Images: {len(images)}\")\n",
                    "    print(f\"Masks: {len(masks)}\")"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Calculate class weights (optional but recommended)\n",
                    "!python class_weights_util.py"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Start training\n",
                    "!python train_mask2former.py --config configs/mask2former_config_kaggle.json"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Download best checkpoint\n",
                    "from IPython.display import FileLink\n",
                    "\n",
                    "checkpoint_path = '/kaggle/working/outputs/mask2former_iris/checkpoints/best.pt'\n",
                    "if os.path.exists(checkpoint_path):\n",
                    "    print(f\"✅ Checkpoint ready for download\")\n",
                    "    FileLink(checkpoint_path)\n",
                    "else:\n",
                    "    print(f\"❌ Checkpoint not found\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Monitor Training\n",
                    "\n",
                    "Watch the training logs above. Key metrics:\n",
                    "- Train Loss: Should decrease steadily\n",
                    "- Val mIoU: Target ≥ 0.90\n",
                    "- Val Dice: Target ≥ 0.93\n",
                    "\n",
                    "Training will stop early if validation metrics don't improve for 15 epochs.\n"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    notebook_path = Path('kaggle_training_notebook.ipynb')
    import json
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✅ Kaggle notebook created: {notebook_path}")

def create_kaggle_readme():
    """Create README for Kaggle setup"""
    
    readme = """# Kaggle Training Setup for Mask2Former Iris Segmentation

## 📋 Prerequisites

1. **Kaggle Account** with phone verification (for GPU access)
2. **Dataset uploaded** as Kaggle dataset
3. **GPU enabled** in notebook settings

## 🚀 Quick Start

### Step 1: Upload Dataset to Kaggle

1. Go to kaggle.com → Datasets → New Dataset
2. Upload your `dataset` folder (with `images/` and `masks/`)
3. Name it (e.g., "iris-segmentation-ubiris")
4. Make it public or private
5. Note the dataset path: `/kaggle/input/your-dataset-name/`

### Step 2: Create New Notebook

1. Kaggle → Code → New Notebook
2. Settings → Accelerator → **GPU P100 or T4**
3. Add your dataset to notebook (+ Add Data → Your Dataset)

### Step 3: Upload Code Files

Upload these files to Kaggle Dataset or use Git:

**Required files:**
```
src/
├── data/
│   ├── dataset.py
│   ├── transforms.py
│   └── __init__.py
├── models/
│   ├── mask2former.py
│   ├── heads.py
│   └── __init__.py
├── losses/
│   ├── mask2former_loss.py
│   └── __init__.py
├── training/
│   ├── mask2former_trainer.py
│   └── __init__.py
└── evaluation/
    └── metrics.py

configs/
└── mask2former_config_kaggle.json

train_mask2former.py
class_weights_util.py (optional)
requirements.txt
```

### Step 4: Update Config

Edit `mask2former_config_kaggle.json`:

```json
{
  "data": {
    "dataset_dir": "/kaggle/input/YOUR-DATASET-NAME/dataset",
    "dataset_root": "/kaggle/input/YOUR-DATASET-NAME/dataset",
    "images_dir": "/kaggle/input/YOUR-DATASET-NAME/dataset/images",
    "masks_dir": "/kaggle/input/YOUR-DATASET-NAME/dataset/masks"
  }
}
```

### Step 5: Run Training

In Kaggle notebook:

```python
# Check GPU
!nvidia-smi

# Install dependencies
!pip install transformers albumentations opencv-python

# Start training
!python train_mask2former.py --config configs/mask2former_config_kaggle.json
```

### Step 6: Download Results

```python
# Download checkpoint
from IPython.display import FileLink
FileLink('/kaggle/working/outputs/mask2former_iris/checkpoints/best.pt')
```

## ⚙️ Kaggle-Optimized Settings

- **Batch Size**: 8 (P100/T4 can handle this)
- **Epochs**: 100 (fits in 12-hour limit)
- **Mixed Precision**: Enabled (faster training)
- **Gradient Accumulation**: Not needed with larger batch
- **Expected Time**: 8-10 hours

## 📊 GPU Specs

| GPU | Memory | Batch Size | Speed |
|-----|--------|------------|-------|
| P100 | 16GB | 8 | ~1.2 hours/epoch |
| T4 | 16GB | 8 | ~1.5 hours/epoch |

## 💡 Tips

1. **Enable Internet** in notebook settings for downloading pretrained model
2. **Save checkpoints** regularly (auto-saved every 20 epochs)
3. **Monitor logs** for NaN losses (reduce LR if occurs)
4. **Download checkpoint** before session expires
5. **Use WandB** (optional) for better monitoring:
   ```python
   !pip install wandb
   # In notebook: wandb login
   # Set use_wandb: true in config
   ```

## 🔧 Troubleshooting

### Out of Memory
- Reduce batch_size to 4
- Enable gradient checkpointing
- Reduce image_size to 384

### Training Too Slow
- Check GPU is enabled (not CPU)
- Verify mixed precision is on
- Reduce logging frequency

### Dataset Not Found
- Check dataset path in config
- Verify dataset is added to notebook
- Check folder structure (images/ and masks/)

## 📈 Expected Results

- **mIoU**: ≥ 0.90 (target)
- **Dice**: ≥ 0.93 (target)
- **Training Time**: 8-10 hours
- **Best Epoch**: Usually around 60-80

## 🎯 Next Steps After Training

1. Download best checkpoint
2. Run inference locally
3. Evaluate on test set
4. Fine-tune if needed

Good luck! 🚀
"""
    
    readme_path = Path('KAGGLE_SETUP.md')
    with open(readme_path, 'w') as f:
        f.write(readme)
    
    print(f"✅ Kaggle README created: {readme_path}")

def main():
    """Main setup function"""
    
    print("=" * 70)
    print("  🚀 KAGGLE TRAINING SETUP")
    print("=" * 70)
    
    # Create files
    create_kaggle_config()
    print()
    create_kaggle_notebook()
    print()
    create_kaggle_readme()
    
    print("\n" + "=" * 70)
    print("  ✅ KAGGLE SETUP COMPLETE!")
    print("=" * 70)
    
    print("\n📝 Files created:")
    print("   1. configs/mask2former_config_kaggle.json")
    print("   2. kaggle_training_notebook.ipynb")
    print("   3. KAGGLE_SETUP.md")
    
    print("\n🎯 Next steps:")
    print("   1. Read KAGGLE_SETUP.md for detailed instructions")
    print("   2. Upload dataset to Kaggle")
    print("   3. Update dataset path in config")
    print("   4. Create Kaggle notebook")
    print("   5. Upload code files")
    print("   6. Run training!")
    
    print("\n💡 Key differences vs local:")
    print("   • Batch size: 8 (vs 2 local)")
    print("   • GPU: P100/T4 16GB (vs RTX 3050 4GB)")
    print("   • Training time: 8-10 hours (vs 5-7 days)")
    print("   • No accumulation needed")
    print("   • Faster convergence")
    
    print("\n⚠️  Remember:")
    print("   • Enable GPU in notebook settings")
    print("   • Download checkpoint before session ends")
    print("   • Kaggle has 12-hour time limit")
    print("   • Save work frequently")

if __name__ == "__main__":
    main()
"""
Configuration management utilities
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Union
import argparse


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML file
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yml', '.yaml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    print(f"Configuration loaded from {config_path}")
    return config


def save_config(config: Dict[str, Any], save_path: Union[str, Path], format: str = 'json'):
    """
    Save configuration to file
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
        format: File format ('json' or 'yaml')
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        if format.lower() == 'json':
            json.dump(config, f, indent=2)
        elif format.lower() in ['yml', 'yaml']:
            yaml.safe_dump(config, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    print(f"Configuration saved to {save_path}")


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values (deep merge)
    
    Args:
        config: Original configuration
        updates: Updates to apply
    
    Returns:
        Updated configuration
    """
    import copy
    
    updated_config = copy.deepcopy(config)
    
    def deep_update(base_dict, update_dict):
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    deep_update(updated_config, updates)
    return updated_config


def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration for iris segmentation training
    
    Returns:
        Default configuration dictionary
    """
    return {
        "project_name": "iris-segmentation-ubiris",
        "run_name": "segformer-b1-enhanced",
        "tags": ["segformer", "iris", "segmentation", "ubiris"],
        "seed": 42,
        
        "data": {
            "dataset_root": "dataset",
            "batch_size": 8,
            "num_workers": 4,
            "use_subject_split": True,
            "preserve_aspect": True,
            "image_size": 512
        },
        
        "model": {
            "model_name": "nvidia/segformer-b1-finetuned-ade-512-512",
            "model_type": "enhanced",
            "num_labels": 2,
            "add_boundary_head": True,
            "freeze_encoder": True,
            "freeze_epochs": 10
        },
        
        "loss": {
            "loss_type": "combined",
            "ce_weight": 0.5,
            "dice_weight": 0.5,
            "boundary_weight": 0.25,
            "aux_weight": 0.2,
            "use_focal": False
        },
        
        "optimizer": {
            "base_lr": 3e-5,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999]
        },
        
        "scheduler": {
            "type": "polynomial",
            "warmup_steps": 1000,
            "power": 0.9
        },
        
        "training": {
            "num_epochs": 160,
            "patience": 15,
            "save_freq": 25,
            "eval_freq": 1,
            "log_freq": 100,
            "gradient_clip": 1.0,
            "mixed_precision": True
        },
        
        "num_classes": 2,
        "class_distribution": [0.93, 0.07],
        "class_names": ["background/pupil", "iris"],
        
        "output_dir": "outputs/segformer_iris",
        
        "evaluation": {
            "metrics": ["pixel_accuracy", "mean_iou", "mean_dice", "boundary_f1"],
            "primary_metric": "mean_iou",
            "save_predictions": True,
            "save_failed_cases": True,
            "iou_threshold": 0.8
        }
    }


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for training
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train SegFormer for iris segmentation')
    
    # Config file
    parser.add_argument('--config', type=str, help='Path to config file')
    
    # Quick setup options
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=160, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--output', type=str, default='outputs/quick_train', help='Output directory')
    
    # Model options
    parser.add_argument('--model', type=str, default='enhanced', 
                       choices=['enhanced', 'deep_supervision'], help='Model type')
    parser.add_argument('--model-name', type=str, 
                       default='nvidia/segformer-b1-finetuned-ade-512-512', 
                       help='HuggingFace model name')
    
    # Training options
    parser.add_argument('--no-wandb', action='store_true', help='Disable Weights & Biases')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--mixed-precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--gradient-clip', type=float, default=1.0, help='Gradient clipping value')
    
    # Data options
    parser.add_argument('--no-subject-split', action='store_true', help='Disable subject-aware splitting')
    parser.add_argument('--no-aspect-preserve', action='store_true', help='Disable aspect ratio preservation')
    parser.add_argument('--image-size', type=int, default=512, help='Input image size')
    
    # Loss options  
    parser.add_argument('--loss-type', type=str, default='combined',
                       choices=['combined', 'focal', 'adaptive', 'unified'], help='Loss function type')
    
    # Resume training
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Create configuration from command line arguments
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        Configuration dictionary
    """
    if args.config:
        config = load_config(args.config)
        
        # Override with command line arguments
        if args.batch_size != 8:
            config['data']['batch_size'] = args.batch_size
        if args.epochs != 160:
            config['training']['num_epochs'] = args.epochs
        if args.lr != 3e-5:
            config['optimizer']['base_lr'] = args.lr
        if args.output != 'outputs/quick_train':
            config['output_dir'] = args.output
            
    else:
        # Create config from arguments
        config = create_default_config()
        config['data']['batch_size'] = args.batch_size
        config['training']['num_epochs'] = args.epochs
        config['optimizer']['base_lr'] = args.lr
        config['output_dir'] = args.output
        config['model']['model_type'] = args.model
        config['model']['model_name'] = args.model_name
        config['loss']['loss_type'] = args.loss_type
        config['data']['use_subject_split'] = not args.no_subject_split
        config['data']['preserve_aspect'] = not args.no_aspect_preserve
        config['data']['image_size'] = args.image_size
        config['training']['gradient_clip'] = args.gradient_clip
        config['training']['mixed_precision'] = args.mixed_precision
    
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration for required fields and sensible values
    
    Args:
        config: Configuration to validate
    
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_fields = [
        'data.dataset_root',
        'data.batch_size',
        'model.model_name',
        'model.num_labels',
        'training.num_epochs',
        'optimizer.base_lr'
    ]
    
    for field in required_fields:
        keys = field.split('.')
        current = config
        for key in keys:
            if key not in current:
                raise ValueError(f"Missing required config field: {field}")
            current = current[key]
    
    # Validate ranges
    if config['data']['batch_size'] <= 0:
        raise ValueError("Batch size must be positive")
    
    if config['training']['num_epochs'] <= 0:
        raise ValueError("Number of epochs must be positive")
    
    if config['optimizer']['base_lr'] <= 0:
        raise ValueError("Learning rate must be positive")
    
    print("✅ Configuration validation passed")
    return True

#!/usr/bin/env python3
"""
Pre-training comprehensive check script
Kiểm tra toàn diện trước khi training Mask2Former
"""

import os
import sys
from pathlib import Path
import torch
import json

def print_section(title):
    """In tiêu đề section"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def check_python_environment():
    """Kiểm tra Python environment"""
    print_section("1️⃣  PYTHON ENVIRONMENT")
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    print(f"✅ Python executable: {sys.executable}")
    
    # Kiểm tra các thư viện quan trọng
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision', 
        'transformers': 'Transformers (HuggingFace)',
        'albumentations': 'Albumentations',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'wandb': 'Weights & Biases (optional)'
    }
    
    missing_packages = []
    installed_packages = {}
    
    for package, name in required_packages.items():
        try:
            if package == 'cv2':
                import cv2
                installed_packages[name] = cv2.__version__
            elif package == 'PIL':
                from PIL import Image
                installed_packages[name] = Image.__version__
            elif package == 'sklearn':
                import sklearn
                installed_packages[name] = sklearn.__version__
            else:
                module = __import__(package)
                installed_packages[name] = module.__version__
        except ImportError:
            missing_packages.append(name)
    
    print(f"\n📦 Installed packages:")
    for name, version in installed_packages.items():
        print(f"   ✅ {name}: {version}")
    
    if missing_packages:
        print(f"\n❌ Missing packages:")
        for name in missing_packages:
            print(f"   ❌ {name}")
        return False
    
    return True

def check_cuda():
    """Kiểm tra CUDA availability"""
    print_section("2️⃣  CUDA & GPU")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ cuDNN version: {torch.backends.cudnn.version()}")
        print(f"✅ Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1e9
            print(f"\n   GPU {i}: {props.name}")
            print(f"      Memory: {memory_gb:.1f} GB")
            print(f"      Compute Capability: {props.major}.{props.minor}")
            
            # Kiểm tra bộ nhớ khả dụng
            if memory_gb < 6:
                print(f"      ⚠️  Low memory - reduce batch_size to 2")
            elif memory_gb < 12:
                print(f"      ⚠️  Medium memory - batch_size max 4")
            else:
                print(f"      ✅ Good memory - batch_size 8+ OK")
        
        return True
    else:
        print(f"⚠️  CUDA not available - will use CPU (very slow)")
        return False

def check_dataset(dataset_root='dataset'):
    """Kiểm tra dataset structure"""
    print_section("3️⃣  DATASET STRUCTURE")
    
    dataset_path = Path(dataset_root)
    images_dir = dataset_path / 'images'
    masks_dir = dataset_path / 'masks'
    
    # Kiểm tra directories
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataset_root}")
        return False
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return False
    
    if not masks_dir.exists():
        print(f"❌ Masks directory not found: {masks_dir}")
        return False
    
    print(f"✅ Dataset directory: {dataset_path}")
    print(f"✅ Images directory: {images_dir}")
    print(f"✅ Masks directory: {masks_dir}")
    
    # Đếm files
    image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(f'*{ext}')))
        image_files.extend(list(images_dir.glob(f'*{ext.upper()}')))
    
    mask_files = []
    for ext in image_extensions:
        mask_files.extend(list(masks_dir.glob(f'*{ext}')))
        mask_files.extend(list(masks_dir.glob(f'*{ext.upper()}')))
    
    print(f"\n📊 File counts:")
    print(f"   Images: {len(image_files)}")
    print(f"   Masks: {len(mask_files)}")
    
    if len(image_files) == 0:
        print(f"❌ No images found!")
        return False
    
    if len(mask_files) == 0:
        print(f"❌ No masks found!")
        return False
    
    # Kiểm tra format
    image_formats = {}
    for f in image_files:
        ext = f.suffix.lower()
        image_formats[ext] = image_formats.get(ext, 0) + 1
    
    print(f"\n📁 Image formats:")
    for ext, count in sorted(image_formats.items()):
        print(f"   {ext}: {count} files")
    
    # Test load 1 sample
    print(f"\n🧪 Testing sample loading...")
    try:
        from PIL import Image
        
        sample_img = Image.open(image_files[0])
        print(f"   ✅ Sample image loaded: {image_files[0].name}")
        print(f"      Size: {sample_img.size}")
        print(f"      Mode: {sample_img.mode}")
        
        sample_mask = Image.open(mask_files[0])
        print(f"   ✅ Sample mask loaded: {mask_files[0].name}")
        print(f"      Size: {sample_mask.size}")
        print(f"      Mode: {sample_mask.mode}")
        
        return True
    except Exception as e:
        print(f"   ❌ Failed to load samples: {e}")
        return False

def check_source_code():
    """Kiểm tra source code structure"""
    print_section("4️⃣  SOURCE CODE STRUCTURE")
    
    required_files = [
        'src/data/dataset.py',
        'src/data/transforms.py',
        'src/models/mask2former.py',
        'src/losses/mask2former_loss.py',
        'src/training/mask2former_trainer.py',
        'src/inference/mask2former_inference.py',
        'configs/mask2former_config.json'
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} NOT FOUND")
            all_exist = False
    
    return all_exist

def check_config(config_path='configs/mask2former_config.json'):
    """Kiểm tra config file"""
    print_section("5️⃣  CONFIGURATION")
    
    if not Path(config_path).exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"✅ Config loaded from: {config_path}")
        
        # Kiểm tra các settings quan trọng
        print(f"\n📋 Key settings:")
        print(f"   Model: {config.get('model', {}).get('architecture', 'N/A')}")
        print(f"   Num labels: {config.get('num_classes', 'N/A')}")
        print(f"   Batch size: {config.get('data', {}).get('batch_size', 'N/A')}")
        print(f"   Image size: {config.get('data', {}).get('image_size', 'N/A')}")
        print(f"   Epochs: {config.get('training', {}).get('num_epochs', 'N/A')}")
        print(f"   Learning rate: {config.get('optimizer', {}).get('base_lr', 'N/A')}")
        print(f"   Output dir: {config.get('output_dir', 'N/A')}")
        
        # Kiểm tra class weights
        loss_config = config.get('loss', {})
        if 'class_weights' in loss_config:
            print(f"   Class weights: {loss_config['class_weights']}")
        else:
            print(f"   ⚠️  No class_weights in config")
        
        return True
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False

def test_dataset_loader(dataset_root='dataset'):
    """Test dataset loader"""
    print_section("6️⃣  DATASET LOADER TEST")
    
    try:
        sys.path.insert(0, 'src')
        from src.data.dataset import UbirisDataset
        
        print(f"Creating dataset...")
        ds = UbirisDataset(dataset_root, split='train', use_subject_split=True)
        
        print(f"✅ Dataset created successfully")
        print(f"   Total samples: {len(ds)}")
        
        print(f"\nLoading first sample...")
        sample = ds[0]
        
        print(f"✅ Sample loaded successfully:")
        print(f"   pixel_values: {sample['pixel_values'].shape}")
        print(f"   labels: {sample['labels'].shape}")
        print(f"   boundary: {sample['boundary'].shape}")
        print(f"   labels unique: {torch.unique(sample['labels']).tolist()}")
        
        # Tính class distribution
        print(f"\nCalculating class distribution from 20 samples...")
        total_pixels = 0
        iris_pixels = 0
        
        for i in range(min(20, len(ds))):
            s = ds[i]
            labels = s['labels']
            total_pixels += labels.numel()
            iris_pixels += (labels == 1).sum().item()
        
        iris_ratio = iris_pixels / total_pixels
        print(f"   Background/Pupil: {(1-iris_ratio)*100:.1f}%")
        print(f"   Iris: {iris_ratio*100:.1f}%")
        
        if iris_ratio < 0.05:
            print(f"   ⚠️  Very low iris ratio - check mask processing")
        elif iris_ratio > 0.4:
            print(f"   ⚠️  Very high iris ratio - check mask processing")
        else:
            print(f"   ✅ Iris ratio looks reasonable")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation"""
    print_section("7️⃣  MODEL CREATION TEST")
    
    try:
        sys.path.insert(0, 'src')
        from src.models.mask2former import create_model
        
        print(f"Creating Mask2Former model...")
        model = create_model(
            architecture='mask2former',
            num_labels=2,
            add_boundary_head=True
        )
        
        print(f"✅ Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        print(f"\nTesting forward pass...")
        dummy_input = torch.randn(1, 3, 512, 512)
        dummy_labels = torch.randint(0, 2, (1, 512, 512))
        
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_input, dummy_labels)
        
        print(f"✅ Forward pass successful")
        print(f"   Output logits shape: {outputs['logits'].shape}")
        if 'boundary_logits' in outputs:
            print(f"   Boundary logits shape: {outputs['boundary_logits'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_class_weights():
    """Check if class weights are calculated"""
    print_section("8️⃣  CLASS WEIGHTS")
    
    weights_file = Path('class_weights.pt')
    
    if weights_file.exists():
        try:
            weights_info = torch.load(weights_file, map_location='cpu', weights_only=False)
            print(f"✅ Class weights file found: {weights_file}")
            print(f"   Weight tensor: {weights_info['weight_tensor']}")
            
            if 'class_counts' in weights_info:
                print(f"   Class counts: {weights_info['class_counts']}")
            
            return True
        except Exception as e:
            print(f"⚠️  Class weights file exists but failed to load: {e}")
            return False
    else:
        print(f"⚠️  No class weights file found")
        print(f"   📝 Run: python class_weights_util.py")
        return False

def print_summary(checks):
    """In tóm tắt kết quả"""
    print_section("📊 SUMMARY")
    
    all_passed = all(checks.values())
    
    print(f"\nCheck results:")
    for name, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}  {name}")
    
    print(f"\n{'='*70}")
    if all_passed:
        print(f"✅ ALL CHECKS PASSED! Ready to start training!")
        print(f"{'='*70}")
        print(f"\n🚀 To start training:")
        print(f"   python train_mask2former.py --config configs/mask2former_config.json")
        print(f"\n💡 Optional flags:")
        print(f"   --wandb          Enable Weights & Biases logging")
        print(f"   --device cuda    Force CUDA device")
        print(f"   --resume PATH    Resume from checkpoint")
    else:
        print(f"❌ SOME CHECKS FAILED - Please fix issues above")
        print(f"{'='*70}")
        
        # Suggest fixes
        print(f"\n💡 Suggested fixes:")
        if not checks.get('Python Environment'):
            print(f"   • Install missing packages: pip install -r requirements.txt")
        if not checks.get('Dataset'):
            print(f"   • Check dataset structure and file formats")
            print(f"   • Run: python verify_tiff_dataset.py")
        if not checks.get('Source Code'):
            print(f"   • Ensure all source files are present")
        if not checks.get('Config'):
            print(f"   • Check config file exists and is valid JSON")
        if not checks.get('Dataset Loader'):
            print(f"   • Debug dataset loading issues")
            print(f"   • Check mask format and values")
        if not checks.get('Model Creation'):
            print(f"   • Check transformers library installation")
            print(f"   • Ensure Mask2Former model is available")
        if not checks.get('Class Weights'):
            print(f"   • Run: python class_weights_util.py")

def main():
    """Main function"""
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║         🔬 MASK2FORMER IRIS SEGMENTATION PRE-TRAINING CHECK       ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    checks = {}
    
    # Run all checks
    checks['Python Environment'] = check_python_environment()
    checks['CUDA'] = check_cuda()
    checks['Dataset'] = check_dataset()
    checks['Source Code'] = check_source_code()
    checks['Config'] = check_config()
    checks['Dataset Loader'] = test_dataset_loader()
    checks['Model Creation'] = test_model_creation()
    checks['Class Weights'] = check_class_weights()
    
    # Print summary
    print_summary(checks)
    
    # Return exit code
    return 0 if all(checks.values()) else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
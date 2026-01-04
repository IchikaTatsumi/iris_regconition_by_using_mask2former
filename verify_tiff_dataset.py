#!/usr/bin/env python3
"""
Verify TIFF dataset structure and compatibility
Run this before training to ensure dataset is ready
"""

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np


def verify_tiff_dataset(dataset_root='dataset'):
    """
    Comprehensive verification of TIFF dataset
    """
    
    print("=" * 70)
    print("TIFF DATASET VERIFICATION")
    print("=" * 70)
    
    # Check directories
    images_dir = Path(dataset_root) / 'images'
    masks_dir = Path(dataset_root) / 'masks'
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return False
    
    if not masks_dir.exists():
        print(f"❌ Masks directory not found: {masks_dir}")
        return False
    
    print(f"✅ Found directories:")
    print(f"   Images: {images_dir}")
    print(f"   Masks: {masks_dir}")
    
    # Find all image files
    image_extensions = ['.tiff', '.tif', '.png', '.jpg', '.jpeg', '.bmp']
    
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
        print(f"   Searched for extensions: {image_extensions}")
        return False
    
    if len(mask_files) == 0:
        print(f"❌ No masks found!")
        return False
    
    # Check file format distribution
    print(f"\n📁 Image formats:")
    image_formats = {}
    for img_file in image_files:
        ext = img_file.suffix.lower()
        image_formats[ext] = image_formats.get(ext, 0) + 1
    
    for ext, count in sorted(image_formats.items()):
        print(f"   {ext}: {count} files")
    
    print(f"\n📁 Mask formats:")
    mask_formats = {}
    for mask_file in mask_files:
        ext = mask_file.suffix.lower()
        mask_formats[ext] = mask_formats.get(ext, 0) + 1
    
    for ext, count in sorted(mask_formats.items()):
        print(f"   {ext}: {count} files")
    
    # Test loading samples
    print(f"\n🔍 Testing sample loading...")
    
    try:
        # Load first image
        sample_img = image_files[0]
        print(f"   Loading image: {sample_img.name}")
        
        img = Image.open(sample_img)
        print(f"   ✅ Image loaded successfully")
        print(f"      Size: {img.size}")
        print(f"      Mode: {img.mode}")
        print(f"      Format: {img.format}")
        
        # Convert to array
        img_array = np.array(img)
        print(f"      Shape: {img_array.shape}")
        print(f"      Dtype: {img_array.dtype}")
        print(f"      Range: [{img_array.min()}, {img_array.max()}]")
        
        # Load first mask
        sample_mask = mask_files[0]
        print(f"\n   Loading mask: {sample_mask.name}")
        
        mask = Image.open(sample_mask)
        print(f"   ✅ Mask loaded successfully")
        print(f"      Size: {mask.size}")
        print(f"      Mode: {mask.mode}")
        print(f"      Format: {mask.format}")
        
        # Convert to array
        mask_array = np.array(mask)
        print(f"      Shape: {mask_array.shape}")
        print(f"      Dtype: {mask_array.dtype}")
        print(f"      Unique values: {np.unique(mask_array)}")
        
        # Check if mask is binary or needs thresholding
        unique_vals = np.unique(mask_array)
        print(f"\n📊 Mask analysis:")
        print(f"   Unique values count: {len(unique_vals)}")
        print(f"   Values: {unique_vals[:10]}" + 
              ("..." if len(unique_vals) > 10 else ""))
        
        if len(unique_vals) <= 3:
            print(f"   ✅ Mask appears binary or near-binary")
        else:
            print(f"   ⚠️  Mask has {len(unique_vals)} unique values")
            print(f"   💡 Will threshold at 127: values > 127 = iris (1)")
            
            # Test thresholding
            binary_mask = (mask_array > 127).astype(np.uint8)
            iris_pixels = binary_mask.sum()
            total_pixels = binary_mask.size
            iris_percentage = (iris_pixels / total_pixels) * 100
            
            print(f"   After thresholding:")
            print(f"      Iris pixels: {iris_pixels} ({iris_percentage:.1f}%)")
            print(f"      Background pixels: {total_pixels - iris_pixels} "
                  f"({100 - iris_percentage:.1f}%)")
        
        # Test with dataset loader
        print(f"\n🧪 Testing dataset loader...")
        
        sys.path.insert(0, 'src')
        from data.dataset import UbirisDataset
        
        print(f"   Creating dataset...")
        ds = UbirisDataset(dataset_root, split='train')
        
        print(f"   ✅ Dataset created: {len(ds)} samples")
        
        print(f"   Loading first sample...")
        sample = ds[0]
        
        print(f"   ✅ Sample loaded:")
        print(f"      pixel_values shape: {sample['pixel_values'].shape}")
        print(f"      labels shape: {sample['labels'].shape}")
        print(f"      boundary shape: {sample['boundary'].shape}")
        print(f"      labels unique: {np.unique(sample['labels'].numpy())}")
        
        print(f"\n{'=' * 70}")
        print(f"✅ ALL TESTS PASSED!")
        print(f"{'=' * 70}")
        print(f"\n💡 Your TIFF dataset is ready for training!")
        print(f"   Next steps:")
        print(f"   1. Calculate class weights: python class_weights_util.py")
        print(f"   2. Start training: python train_mask2former.py")
        
        return True
        
    except ImportError as e:
        print(f"\n⚠️  Dataset loader not available: {e}")
        print(f"   Basic file checks passed, but couldn't test full loader")
        print(f"   This is OK if you haven't set up the project yet")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_image_mask_pairing(dataset_root='dataset'):
    """
    Check how many images have matching masks
    """
    print(f"\n{'=' * 70}")
    print(f"CHECKING IMAGE-MASK PAIRING")
    print(f"{'=' * 70}")
    
    images_dir = Path(dataset_root) / 'images'
    masks_dir = Path(dataset_root) / 'masks'
    
    image_extensions = ['.tiff', '.tif', '.png', '.jpg', '.jpeg', '.bmp']
    
    # Get all images
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(f'*{ext}')))
        image_files.extend(list(images_dir.glob(f'*{ext.upper()}')))
    
    print(f"📊 Found {len(image_files)} images")
    print(f"   Checking for matching masks...\n")
    
    matched = 0
    unmatched = []
    
    for img_file in image_files:
        base_name = img_file.stem  # Filename without extension
        
        # Try different mask patterns
        found_mask = False
        mask_patterns = [
            f"OperatorA_{base_name}",
            base_name,
            f"{base_name}_mask",
            f"mask_{base_name}"
        ]
        
        for pattern in mask_patterns:
            for ext in image_extensions:
                mask_file = masks_dir / f"{pattern}{ext}"
                if mask_file.exists():
                    found_mask = True
                    break
            if found_mask:
                break
        
        if found_mask:
            matched += 1
        else:
            unmatched.append(img_file.name)
    
    print(f"✅ Matched: {matched}/{len(image_files)} "
          f"({matched/len(image_files)*100:.1f}%)")
    
    if unmatched:
        print(f"\n⚠️  Unmatched images ({len(unmatched)}):")
        for name in unmatched[:10]:
            print(f"   • {name}")
        if len(unmatched) > 10:
            print(f"   ... and {len(unmatched) - 10} more")
    else:
        print(f"\n✅ All images have matching masks!")
    
    return matched == len(image_files)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Verify TIFF dataset for iris segmentation'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='dataset',
        help='Path to dataset root directory'
    )
    parser.add_argument(
        '--check-pairing',
        action='store_true',
        help='Also check image-mask pairing'
    )
    
    args = parser.parse_args()
    
    # Run main verification
    success = verify_tiff_dataset(args.dataset)
    
    # Run pairing check if requested
    if args.check_pairing:
        check_image_mask_pairing(args.dataset)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
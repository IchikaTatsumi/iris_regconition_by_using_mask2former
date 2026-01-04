#!/usr/bin/env python3
"""
Debug script to verify data pipeline fixes
"""

import sys
import os
sys.path.append('src')

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from data.dataset import UbirisDataset

def check_mask_values(dataset_root='dataset'):
    """Check for anti-aliasing issues in masks"""
    print("🔍 Checking mask values for anti-aliasing issues...")
    
    ds = UbirisDataset(dataset_root, split='train', use_subject_split=False)
    
    problematic_masks = 0
    total_checked = min(50, len(ds))  # Check first 50 samples
    
    for i in range(total_checked):
        try:
            sample = ds[i]
            # The debug print in dataset.py will show problematic masks
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
    
    print(f"✅ Checked {total_checked} mask files")

def check_label_distribution(dataset_root='dataset'):
    """Check class distribution in processed labels"""
    print("\n📊 Checking label distribution...")
    
    ds = UbirisDataset(dataset_root, split='train')
    
    total_pixels = 0
    iris_pixels = 0
    
    for i in range(min(20, len(ds))):  # Check 20 samples
        sample = ds[i]
        labels = sample['labels']
        
        total_pixels += labels.numel()
        iris_pixels += (labels == 1).sum().item()
        
        # Check for unexpected values
        unique_vals = torch.unique(labels)
        if not torch.equal(unique_vals, torch.tensor([0, 1])) and not torch.equal(unique_vals, torch.tensor([0])) and not torch.equal(unique_vals, torch.tensor([1])):
            print(f"⚠️  Unexpected label values in sample {i}: {unique_vals}")
    
    iris_ratio = iris_pixels / total_pixels if total_pixels > 0 else 0
    print(f"📈 Class distribution (from {min(20, len(ds))} samples):")
    print(f"   Background/Pupil: {(1-iris_ratio)*100:.1f}%")
    print(f"   Iris: {iris_ratio*100:.1f}%")
    
    if iris_ratio < 0.05:
        print("⚠️  Very low iris ratio - this explains the 'all background' predictions!")
    elif iris_ratio > 0.4:
        print("⚠️  Unusually high iris ratio - check mask processing")
    else:
        print("✅ Iris ratio looks reasonable")

def visualize_sample(dataset_root='dataset', sample_idx=0):
    """Visualize a sample to check alignment"""
    print(f"\n🖼️  Visualizing sample {sample_idx}...")
    
    ds = UbirisDataset(dataset_root, split='train')
    sample = ds[sample_idx]
    
    # Denormalize image
    img_tensor = sample['pixel_values']
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_denorm = (img_tensor * std + mean).clamp(0, 1)
    img_np = img_denorm.permute(1, 2, 0).numpy()
    
    # Get mask and boundary
    mask_np = sample['labels'].numpy()
    boundary_np = sample['boundary'].numpy()
    
    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(img_np)
    axes[0].set_title('Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title(f'Mask (unique: {np.unique(mask_np)})')
    axes[1].axis('off')
    
    axes[2].imshow(boundary_np, cmap='gray')
    axes[2].set_title(f'Boundary (unique: {np.unique(boundary_np)})')
    axes[2].axis('off')
    
    # Overlay
    axes[3].imshow(img_np)
    axes[3].imshow(mask_np, alpha=0.3, cmap='Reds')
    axes[3].set_title('Image + Mask Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('debug_sample_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📁 Visualization saved to: debug_sample_visualization.png")
    print(f"   Image shape: {img_tensor.shape}")
    print(f"   Mask shape: {sample['labels'].shape}")
    print(f"   Boundary shape: {sample['boundary'].shape}")

def test_augmentation_consistency(dataset_root='dataset'):
    """Test that augmentations are applied consistently"""
    print("\n🔄 Testing augmentation consistency...")
    
    ds = UbirisDataset(dataset_root, split='train')  # Training = augmentations on
    
    # Get one sample multiple times to see different augmentations
    samples = []
    for _ in range(5):
        sample = ds[0]  # Same index, different augmentations
        samples.append(sample)
    
    # Check that masks and boundaries have same unique values
    for i, sample in enumerate(samples):
        mask_unique = torch.unique(sample['labels']).tolist()
        boundary_unique = torch.unique(sample['boundary']).tolist()
        print(f"   Sample {i}: mask={mask_unique}, boundary={boundary_unique}")
    
    print("✅ If all samples show {0,1} for masks and {0.0,1.0} for boundaries, consistency looks good")

if __name__ == "__main__":
    print("🚀 Starting data pipeline debugging...")
    
    try:
        check_mask_values()
        check_label_distribution()
        visualize_sample()
        test_augmentation_consistency()
        
        print("\n✅ Debugging complete! Check the output for any warnings.")
        print("📋 Next steps:")
        print("   1. If you see anti-aliasing warnings, the threshold fix should help")
        print("   2. If iris ratio is very low (<5%), consider class balancing") 
        print("   3. Check debug_sample_visualization.png to verify mask alignment")
        
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()

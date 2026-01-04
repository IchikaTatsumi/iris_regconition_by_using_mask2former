#!/usr/bin/env python3
"""
Utility to calculate class weights for iris segmentation
"""

import sys
import os
sys.path.append('src')

import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from data.dataset import UbirisDataset

def calculate_class_weights(dataset_root='dataset', max_samples=500):
    """
    Calculate class weights to handle iris/background imbalance
    
    Args:
        dataset_root: Path to dataset
        max_samples: Maximum samples to analyze (for speed)
    
    Returns:
        Dictionary with class weights
    """
    print(f"📊 Calculating class weights from {max_samples} samples...")
    
    ds = UbirisDataset(dataset_root, split='train')
    
    all_labels = []
    total_samples = min(max_samples, len(ds))
    
    for i in range(total_samples):
        if i % 100 == 0:
            print(f"   Processed {i}/{total_samples} samples...")
            
        sample = ds[i]
        labels = sample['labels'].numpy().flatten()
        all_labels.extend(labels)
    
    # Convert to numpy array
    all_labels = np.array(all_labels)
    unique_classes = np.unique(all_labels)
    
    # Calculate weights using sklearn
    class_weights = compute_class_weight(
        'balanced',
        classes=unique_classes,
        y=all_labels
    )
    
    # Create weight dictionary
    weight_dict = {int(cls): float(weight) for cls, weight in zip(unique_classes, class_weights)}
    
    # Calculate statistics
    class_counts = {cls: np.sum(all_labels == cls) for cls in unique_classes}
    total_pixels = len(all_labels)
    
    print("\n📈 Class Distribution:")
    for cls in unique_classes:
        count = class_counts[cls]
        percentage = (count / total_pixels) * 100
        print(f"   Class {cls}: {count:,} pixels ({percentage:.2f}%)")
    
    print(f"\n⚖️  Calculated Weights:")
    for cls, weight in weight_dict.items():
        print(f"   Class {cls}: {weight:.4f}")
    
    # Convert to torch tensor format for loss functions
    weight_tensor = torch.tensor([weight_dict[i] for i in sorted(weight_dict.keys())], dtype=torch.float32)
    
    print(f"\n🔥 PyTorch weight tensor: {weight_tensor}")
    print(f"   Usage: CrossEntropyLoss(weight={weight_tensor})")
    
    return {
        'class_weights_dict': weight_dict,
        'weight_tensor': weight_tensor,
        'class_counts': class_counts
    }

def suggest_loss_functions(weights_info):
    """Suggest appropriate loss functions for the class imbalance"""
    
    weight_tensor = weights_info['weight_tensor']
    max_weight = weight_tensor.max().item()
    
    print(f"\n💡 Recommended Loss Functions:")
    
    print(f"\n1. 🎯 Weighted Cross-Entropy Loss:")
    print(f"   ```python")
    print(f"   import torch.nn as nn")
    print(f"   weights = {weight_tensor}")
    print(f"   criterion = nn.CrossEntropyLoss(weight=weights)")
    print(f"   ```")
    
    print(f"\n2. 🎲 Focal Loss (handles hard examples):")
    print(f"   ```python")
    print(f"   # Focal Loss implementation needed")
    print(f"   criterion = FocalLoss(alpha=0.25, gamma=2.0)")
    print(f"   ```")
    
    print(f"\n3. 🎪 Dice Loss (good for segmentation):")
    print(f"   ```python")
    print(f"   # Dice Loss implementation needed")
    print(f"   criterion = DiceLoss(smooth=1.0)")
    print(f"   ```")
    
    if max_weight > 10:
        print(f"\n⚠️  Weight ratio is high ({max_weight:.1f}x). Consider:")
        print(f"   • Combining Dice + Weighted CE loss")
        print(f"   • Using Focal loss with gamma=2")
        print(f"   • Data augmentation focused on iris regions")

if __name__ == "__main__":
    try:
        weights_info = calculate_class_weights()
        suggest_loss_functions(weights_info)
        
        # Save weights for training script
        torch.save(weights_info, 'class_weights.pt')
        print(f"\n💾 Saved weights to: class_weights.pt")
        
    except Exception as e:
        print(f"❌ Error calculating weights: {e}")
        import traceback
        traceback.print_exc()

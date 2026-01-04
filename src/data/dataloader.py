"""
DataLoader utilities for UBIRIS V2 dataset
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .dataset import UbirisDataset


def get_transforms(augmentation=False):
    """
    Get image and mask transforms for training/validation
    
    Args:
        augmentation: Whether to apply data augmentation
    
    Returns:
        Tuple of (image_transform, mask_transform)
    """
    if augmentation:
        # Training transforms with augmentation
        image_transform = transforms.Compose([
            transforms.Resize((300, 400)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        mask_transform = transforms.Compose([
            transforms.Resize((300, 400), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomHorizontalFlip(p=0.5)
        ])
    else:
        # Validation/test transforms without augmentation
        image_transform = transforms.Compose([
            transforms.Resize((300, 400)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        mask_transform = transforms.Compose([
            transforms.Resize((300, 400), interpolation=transforms.InterpolationMode.NEAREST)
        ])
    
    return image_transform, mask_transform


def create_dataloaders(dataset_root, batch_size=8, num_workers=4, pin_memory=True):
    """
    Create DataLoaders for train, validation, and test splits
    
    Args:
        dataset_root: Path to dataset root directory
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        Dictionary with train, val, and test DataLoaders
    """
    
    # Get transforms
    train_img_transform, train_mask_transform = get_transforms(augmentation=True)
    val_img_transform, val_mask_transform = get_transforms(augmentation=False)
    
    # Create datasets
    train_dataset = UbirisDataset(
        dataset_root=dataset_root,
        split='train',
        transform=train_img_transform,
        mask_transform=train_mask_transform
    )
    
    val_dataset = UbirisDataset(
        dataset_root=dataset_root,
        split='val',
        transform=val_img_transform,
        mask_transform=val_mask_transform
    )
    
    test_dataset = UbirisDataset(
        dataset_root=dataset_root,
        split='test',
        transform=val_img_transform,
        mask_transform=val_mask_transform
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"DataLoaders created:")
    print(f"  Train: {len(train_loader)} batches, {len(train_dataset)} samples")
    print(f"  Val: {len(val_loader)} batches, {len(val_dataset)} samples")
    print(f"  Test: {len(test_loader)} batches, {len(test_dataset)} samples")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def collate_fn(batch):
    """
    Custom collate function for SegFormer training
    
    Args:
        batch: List of samples from dataset
    
    Returns:
        Batched data suitable for SegFormer
    """
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'pixel_values': pixel_values,
        'labels': labels
    }


def get_segformer_dataloaders(dataset_root, batch_size=8, num_workers=4):
    """
    Create DataLoaders specifically formatted for SegFormer training
    
    Args:
        dataset_root: Path to dataset root directory
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes
    
    Returns:
        Dictionary with train and val DataLoaders for SegFormer
    """
    
    # Create datasets with SegFormer-compatible transforms
    train_img_transform = transforms.Compose([
        transforms.Resize((512, 512)),  # SegFormer typically uses 512x512
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_mask_transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.RandomHorizontalFlip(p=0.5)
    ])
    
    val_img_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_mask_transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST)
    ])
    
    # Create datasets
    train_dataset = UbirisDataset(
        dataset_root=dataset_root,
        split='train',
        transform=train_img_transform,
        mask_transform=train_mask_transform
    )
    
    val_dataset = UbirisDataset(
        dataset_root=dataset_root,
        split='val',
        transform=val_img_transform,
        mask_transform=val_mask_transform
    )
    
    # Create DataLoaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"SegFormer DataLoaders created:")
    print(f"  Train: {len(train_loader)} batches, {len(train_dataset)} samples")
    print(f"  Val: {len(val_loader)} batches, {len(val_dataset)} samples")
    print(f"  Classes: 2 (0=background/pupil, 1=iris)")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'num_classes': 2,
        'class_names': ['background/pupil', 'iris']
    }

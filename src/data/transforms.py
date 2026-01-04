"""
Advanced augmentation pipeline for iris segmentation using Albumentations
"""

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("⚠️  Albumentations not available. Install with: pip install albumentations")

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


class AspectRatioPreservingResize:
    """Resize while preserving aspect ratio with padding"""
    
    def __init__(self, target_size=512, padding_mode='constant', padding_value=0):
        self.target_size = target_size
        self.padding_mode = padding_mode
        self.padding_value = padding_value
    
    def __call__(self, image, mask=None):
        h, w = image.shape[:2]
        
        # Calculate scale factor to fit within target size
        scale = min(self.target_size / h, self.target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image and mask
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Calculate padding
        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Apply padding
        image = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=self.padding_value
        )
        
        if mask is not None:
            mask = cv2.copyMakeBorder(
                mask, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=0
            )
            return image, mask
        
        return image


def get_training_transforms(image_size=512, p_flip=0.5):
    """
    Get training transforms with eye-safe augmentations
    
    Args:
        image_size: Target image size
        p_flip: Probability of horizontal flip
    
    Returns:
        Albumentations transform pipeline or fallback torchvision transforms
    """
    if not ALBUMENTATIONS_AVAILABLE:
        # Fallback to basic torchvision transforms
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=p_flip),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return A.Compose([
        # Geometric transforms (mild to preserve iris structure)
        A.HorizontalFlip(p=p_flip),
        # A.VerticalFlip(p=0.1),  # REMOVED: Unrealistic for eye images
        A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT, p=0.3),
        A.RandomScale(scale_limit=0.1, p=0.3),  # 0.9-1.1 scale
        
        # Ensure fixed size after random scale
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
        
        # Color augmentation (simulate lighting conditions)
        A.RandomBrightnessContrast(
            brightness_limit=0.25, 
            contrast_limit=0.25, 
            p=0.4
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.3
        ),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
        
        # Blur effects (simulate focus issues)
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        
        # Noise (reduced intensity for better training)
        A.GaussNoise(var_limit=20.0, p=0.1),
        
        # Normalization
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ], additional_targets={'boundary': 'mask'})


def get_validation_transforms(image_size=512):
    """
    Get validation transforms (no augmentation)
    
    Args:
        image_size: Target image size
    
    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ], additional_targets={'boundary': 'mask'})


def get_inference_transform(image_size=512):
    """
    Get inference transforms (same as validation but simpler)
    
    Args:
        image_size: Target image size
    
    Returns:
        Transform function for inference
    """
    if not ALBUMENTATIONS_AVAILABLE:
        # Fallback to torchvision
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    transform = A.Compose([
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    def inference_transform(image):
        """Apply transform to PIL image"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        transformed = transform(image=image)
        return transformed['image']
    
    return inference_transform


def create_boundary_mask(mask, dilation_size=3):
    """
    Create boundary mask for boundary-aware loss
    
    Args:
        mask: Binary mask (H, W) with values {0, 1}
        dilation_size: Size of morphological kernel
    
    Returns:
        Boundary mask (H, W) with boundary pixels = 1
    """
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy().astype(np.uint8)
    else:
        mask_np = mask.astype(np.uint8)
    
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (dilation_size, dilation_size)
    )
    
    # Create boundary by dilation - erosion
    dilated = cv2.dilate(mask_np, kernel, iterations=1)
    eroded = cv2.erode(mask_np, kernel, iterations=1)
    boundary = dilated ^ eroded  # XOR to get boundary
    
    return torch.from_numpy(boundary) if isinstance(mask, torch.Tensor) else boundary


class IrisAugmentation:
    """
    Iris-specific augmentation pipeline
    """
    
    def __init__(self, image_size=512, training=True, preserve_aspect=True):
        self.image_size = image_size
        self.training = training
        self.preserve_aspect = preserve_aspect
        
        if preserve_aspect:
            self.resize_fn = AspectRatioPreservingResize(image_size)
        
        if training:
            self.transform = get_training_transforms(image_size)
        else:
            self.transform = get_validation_transforms(image_size)
    
    def __call__(self, image, mask):
        """
        Apply transforms to image and mask
        
        Args:
            image: PIL Image or numpy array
            mask: PIL Image or numpy array
        
        Returns:
            Tuple of transformed (image_tensor, mask_tensor, boundary_tensor)
        """
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        
        # Preserve aspect ratio if requested
        if self.preserve_aspect:
            image, mask = self.resize_fn(image, mask)
        else:
            # Standard resize (may distort)
            image = cv2.resize(image, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size), 
                            interpolation=cv2.INTER_NEAREST)
        
        # Create boundary mask before augmentation
        boundary = create_boundary_mask(mask)
        
        # Apply augmentations
        transformed = self.transform(image=image, mask=mask)
        image_tensor = transformed['image']
        mask_tensor = transformed['mask']
        
        # Apply augmentations with boundary as additional target
        # This ensures boundary gets SAME transforms as image/mask
        transformed = self.transform(
            image=image, 
            mask=mask,
            boundary=boundary.astype(np.uint8)
        )
        image_tensor = transformed['image']
        mask_tensor = transformed['mask']
        boundary_tensor = transformed['boundary']
        
        # Ensure correct tensor shapes and types
        mask_tensor = mask_tensor.long().squeeze()
        boundary_tensor = boundary_tensor.float().squeeze()
        
        return image_tensor, mask_tensor, boundary_tensor


def visualize_augmentations(image, mask, num_samples=4, save_path='augmentation_samples.png'):
    """
    Visualize augmentation results
    
    Args:
        image: Input image (PIL or numpy)
        mask: Input mask (PIL or numpy)
        num_samples: Number of augmented samples to show
        save_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    aug = IrisAugmentation(training=True, preserve_aspect=True)
    
    fig, axes = plt.subplots(3, num_samples + 1, figsize=(4 * (num_samples + 1), 12))
    
    # Original
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(mask, cmap='gray')
    axes[1, 0].set_title('Original Mask')
    axes[1, 0].axis('off')
    
    boundary_orig = create_boundary_mask(np.array(mask))
    axes[2, 0].imshow(boundary_orig, cmap='gray')
    axes[2, 0].set_title('Original Boundary')
    axes[2, 0].axis('off')
    
    # Augmented samples
    for i in range(num_samples):
        img_tensor, mask_tensor, boundary_tensor = aug(image, mask)
        
        # Denormalize image
        img_np = img_tensor.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        
        axes[0, i + 1].imshow(img_np)
        axes[0, i + 1].set_title(f'Augmented {i + 1}')
        axes[0, i + 1].axis('off')
        
        axes[1, i + 1].imshow(mask_tensor.numpy(), cmap='gray')
        axes[1, i + 1].set_title(f'Mask {i + 1}')
        axes[1, i + 1].axis('off')
        
        axes[2, i + 1].imshow(boundary_tensor.numpy(), cmap='gray')
        axes[2, i + 1].set_title(f'Boundary {i + 1}')
        axes[2, i + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Augmentation visualization saved to {save_path}")


if __name__ == "__main__":
    # Test the augmentation pipeline
    from PIL import Image
    
    # Load a sample image and mask
    image_path = "dataset/images/C1_S1_I1.png"
    mask_path = "dataset/masks/OperatorA_C1_S1_I1.png"
    
    try:
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Test augmentation
        aug = IrisAugmentation(training=True, preserve_aspect=True)
        img_tensor, mask_tensor, boundary_tensor = aug(image, mask)
        
        print(f"Image tensor shape: {img_tensor.shape}")
        print(f"Mask tensor shape: {mask_tensor.shape}")
        print(f"Boundary tensor shape: {boundary_tensor.shape}")
        print(f"Unique mask values: {torch.unique(mask_tensor)}")
        print(f"Unique boundary values: {torch.unique(boundary_tensor)}")
        
        # Visualize
        visualize_augmentations(image, mask)
        
    except FileNotFoundError:
        print("Sample files not found. Please ensure dataset is available.")

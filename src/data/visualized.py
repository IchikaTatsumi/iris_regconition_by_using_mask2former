"""
Comprehensive visualization of all augmentations used in iris segmentation
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
from PIL import Image
import torch

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("⚠️  Albumentations not available. Install with: pip install albumentations")

from data.transforms import IrisAugmentation, create_boundary_mask


class AugmentationVisualizer:
    """Visualize individual and combined augmentations"""
    
    def __init__(self, image_size=512):
        self.image_size = image_size
        
    def denormalize_tensor(self, tensor):
        """Denormalize tensor for visualization"""
        if tensor.dim() == 3:  # C, H, W
            img_np = tensor.permute(1, 2, 0).numpy()
        else:
            img_np = tensor.numpy()
            
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        return np.clip(img_np, 0, 1)
    
    def visualize_individual_augmentations(self, image, mask, save_path='individual_augmentations.png'):
        """Visualize each augmentation type individually"""
        if not ALBUMENTATIONS_AVAILABLE:
            print("Albumentations not available for individual visualization")
            return
            
        # Convert inputs
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
            
        # Define individual transforms
        transforms_dict = {
            'Original': A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            'Horizontal Flip': A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            'Vertical Flip': A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            'Rotation (10°)': A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Rotate(limit=10, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            'Random Scale': A.Compose([
                A.RandomScale(scale_limit=0.1, p=1.0),
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            'Brightness/Contrast': A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            'Hue/Saturation': A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            'CLAHE': A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            'Gaussian Blur': A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            'Motion Blur': A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.MotionBlur(blur_limit=3, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            'Gaussian Noise': A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        }
        
        # Create visualization
        n_transforms = len(transforms_dict)
        cols = 4
        rows = (n_transforms + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, (name, transform) in enumerate(transforms_dict.items()):
            transformed = transform(image=image, mask=mask)
            img_tensor = transformed['image']
            mask_tensor = transformed['mask']
            
            # Denormalize and visualize
            if name == 'Original':
                img_display = self.denormalize_tensor(img_tensor)
            else:
                img_display = self.denormalize_tensor(img_tensor)
            
            # Create overlay
            overlay = img_display.copy()
            if len(mask_tensor.shape) == 3:
                mask_np = mask_tensor[0].numpy()
            else:
                mask_np = mask_tensor.numpy()
            
            # Add mask overlay in red
            overlay[:, :, 0] = np.where(mask_np > 0.5, 
                                      np.minimum(overlay[:, :, 0] + 0.3, 1.0), 
                                      overlay[:, :, 0])
            
            axes[i].imshow(overlay)
            axes[i].set_title(name, fontsize=10, fontweight='bold')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(transforms_dict), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Individual augmentations saved to {save_path}")
    
    def visualize_progressive_augmentation(self, image, mask, save_path='progressive_augmentation.png'):
        """Show progressive application of augmentations"""
        if not ALBUMENTATIONS_AVAILABLE:
            print("Albumentations not available for progressive visualization")
            return
            
        # Convert inputs
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        
        # Define progressive transforms
        stages = [
            ('Original', A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])),
            ('+ Geometry', A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.3),
                A.RandomScale(scale_limit=0.1, p=0.3),
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])),
            ('+ Color', A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.3),
                A.RandomScale(scale_limit=0.1, p=0.3),
                A.Resize(self.image_size, self.image_size),
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.4),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])),
            ('+ Enhancement', A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.3),
                A.RandomScale(scale_limit=0.1, p=0.3),
                A.Resize(self.image_size, self.image_size),
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.4),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])),
            ('+ Blur/Noise', A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.3),
                A.RandomScale(scale_limit=0.1, p=0.3),
                A.Resize(self.image_size, self.image_size),
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.4),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
                A.OneOf([A.GaussianBlur(blur_limit=3, p=1.0), A.MotionBlur(blur_limit=3, p=1.0)], p=0.2),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]))
        ]
        
        fig, axes = plt.subplots(2, len(stages), figsize=(4 * len(stages), 8))
        
        for i, (name, transform) in enumerate(stages):
            # Generate multiple samples to show variation
            for sample in range(3):  # Try 3 times to get visible changes
                transformed = transform(image=image, mask=mask)
                img_tensor = transformed['image']
                mask_tensor = transformed['mask']
                
                img_display = self.denormalize_tensor(img_tensor)
                
                if len(mask_tensor.shape) == 3:
                    mask_np = mask_tensor[0].numpy()
                else:
                    mask_np = mask_tensor.numpy()
                
                # If we got some change or it's the original, use it
                if sample == 0 or name == 'Original' or not np.allclose(img_display, prev_img if 'prev_img' in locals() else img_display):
                    break
                
            prev_img = img_display.copy()
            
            # Show image
            axes[0, i].imshow(img_display)
            axes[0, i].set_title(f'{name}\nImage', fontsize=10, fontweight='bold')
            axes[0, i].axis('off')
            
            # Show mask
            axes[1, i].imshow(mask_np, cmap='gray')
            axes[1, i].set_title(f'{name}\nMask', fontsize=10, fontweight='bold')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Progressive augmentation saved to {save_path}")
    
    def visualize_boundary_detection(self, image, mask, save_path='boundary_visualization.png'):
        """Visualize boundary detection process"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        
        # Test different dilation sizes
        dilation_sizes = [1, 3, 5, 7]
        
        fig, axes = plt.subplots(2, len(dilation_sizes) + 1, figsize=(4 * (len(dilation_sizes) + 1), 8))
        
        # Original
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(mask, cmap='gray')
        axes[1, 0].set_title('Original Mask')
        axes[1, 0].axis('off')
        
        # Different boundary sizes
        for i, dilation_size in enumerate(dilation_sizes):
            boundary = create_boundary_mask(mask, dilation_size)
            
            # Create overlay
            overlay = image.copy() if len(image.shape) == 3 else np.stack([image]*3, axis=-1)
            if len(overlay.shape) == 3 and overlay.max() <= 1:
                overlay = (overlay * 255).astype(np.uint8)
            
            # Add boundary in red
            boundary_coords = np.where(boundary > 0)
            if len(boundary_coords[0]) > 0:
                overlay[boundary_coords] = [255, 0, 0]
            
            axes[0, i + 1].imshow(overlay)
            axes[0, i + 1].set_title(f'Boundary\n(dilation={dilation_size})')
            axes[0, i + 1].axis('off')
            
            axes[1, i + 1].imshow(boundary, cmap='gray')
            axes[1, i + 1].set_title(f'Boundary Mask\n(dilation={dilation_size})')
            axes[1, i + 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Boundary visualization saved to {save_path}")
    
    def visualize_aspect_ratio_preservation(self, image, mask, save_path='aspect_ratio_comparison.png'):
        """Compare aspect ratio preservation vs standard resize"""
        from data.transforms import AspectRatioPreservingResize
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        
        original_shape = image.shape[:2]
        
        # Standard resize (may distort)
        resized_std = cv2.resize(image, (self.image_size, self.image_size))
        mask_std = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        # Aspect ratio preserving resize
        aspect_resizer = AspectRatioPreservingResize(self.image_size)
        resized_aspect, mask_aspect = aspect_resizer(image, mask)
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # Original
        axes[0, 0].imshow(image)
        axes[0, 0].set_title(f'Original\n{original_shape[1]}×{original_shape[0]}')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(mask, cmap='gray')
        axes[1, 0].set_title('Original Mask')
        axes[1, 0].axis('off')
        
        # Standard resize
        axes[0, 1].imshow(resized_std)
        axes[0, 1].set_title(f'Standard Resize\n{self.image_size}×{self.image_size}\n(may distort)')
        axes[0, 1].axis('off')
        
        axes[1, 1].imshow(mask_std, cmap='gray')
        axes[1, 1].set_title('Standard Mask')
        axes[1, 1].axis('off')
        
        # Aspect preserving
        axes[0, 2].imshow(resized_aspect)
        axes[0, 2].set_title(f'Aspect Preserving\n{self.image_size}×{self.image_size}\n(with padding)')
        axes[0, 2].axis('off')
        
        axes[1, 2].imshow(mask_aspect, cmap='gray')
        axes[1, 2].set_title('Aspect Preserving Mask')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Aspect ratio comparison saved to {save_path}")
    
    def create_comprehensive_visualization(self, image_path, mask_path, output_dir='augmentation_visualizations'):
        """Create comprehensive visualization of all augmentations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load image and mask
        try:
            image = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            return
        
        print(f"Creating visualizations for:")
        print(f"  Image: {image_path}")
        print(f"  Mask: {mask_path}")
        print(f"  Output: {output_dir}/")
        
        # Create all visualizations
        self.visualize_individual_augmentations(
            image, mask, os.path.join(output_dir, 'individual_augmentations.png')
        )
        
        self.visualize_progressive_augmentation(
            image, mask, os.path.join(output_dir, 'progressive_augmentation.png')
        )
        
        self.visualize_boundary_detection(
            image, mask, os.path.join(output_dir, 'boundary_visualization.png')
        )
        
        self.visualize_aspect_ratio_preservation(
            image, mask, os.path.join(output_dir, 'aspect_ratio_comparison.png')
        )
        
        # Use the existing function for full pipeline
        from data.transforms import visualize_augmentations
        visualize_augmentations(
            image, mask, num_samples=6, 
            save_path=os.path.join(output_dir, 'full_pipeline_samples.png')
        )
        
        print(f"\n✅ All visualizations created in {output_dir}/")
        print("Files created:")
        for filename in os.listdir(output_dir):
            if filename.endswith('.png'):
                print(f"  - {filename}")


def main():
    """Main function to create visualizations"""
    visualizer = AugmentationVisualizer(image_size=512)
    
    # Default paths - adjust as needed
    image_path = "dataset/images/C1_S1_I1.png"
    mask_path = "dataset/masks/OperatorA_C1_S1_I1.png"
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("Available images in dataset/images/:")
        try:
            images = [f for f in os.listdir("dataset/images/") if f.endswith(('.png', '.jpg', '.jpeg'))]
            for img in images[:5]:  # Show first 5
                print(f"  - {img}")
            if len(images) > 5:
                print(f"  ... and {len(images) - 5} more")
            image_path = f"dataset/images/{images[0]}" if images else None
        except:
            print("  (directory not accessible)")
    
    if not os.path.exists(mask_path):
        print(f"❌ Mask not found: {mask_path}")
        print("Available masks in dataset/masks/:")
        try:
            masks = [f for f in os.listdir("dataset/masks/") if f.endswith(('.png', '.jpg', '.jpeg'))]
            for mask in masks[:5]:  # Show first 5
                print(f"  - {mask}")
            if len(masks) > 5:
                print(f"  ... and {len(masks) - 5} more")
            mask_path = f"dataset/masks/{masks[0]}" if masks else None
        except:
            print("  (directory not accessible)")
    
    if image_path and mask_path and os.path.exists(image_path) and os.path.exists(mask_path):
        visualizer.create_comprehensive_visualization(image_path, mask_path)
    else:
        print("\n❌ Cannot proceed without valid image and mask files")
        print("Usage: python visualized.py")
        print("Make sure your dataset is structured as:")
        print("  dataset/")
        print("    images/")
        print("    masks/")


if __name__ == "__main__":
    main()

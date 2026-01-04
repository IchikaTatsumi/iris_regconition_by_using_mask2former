"""
COMPLETE FIXED dataset.py with TIFF support
File: src/data/dataset.py
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import re
from sklearn.model_selection import train_test_split

try:
    from data.transforms import IrisAugmentation, create_boundary_mask
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from transforms import IrisAugmentation, create_boundary_mask


# FIXED: Support multiple image formats including TIFF
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']


class UbirisDataset(Dataset):
    """
    FIXED: UBIRIS V2 Dataset for iris segmentation with TIFF support
    """
    
    def __init__(
        self, 
        dataset_root, 
        split='train', 
        transform=None, 
        mask_transform=None, 
        use_subject_split=True, 
        preserve_aspect=True, 
        image_size=512, 
        seed=42
    ):
        """
        Args:
            dataset_root: Path to dataset root (contains 'images' and 'masks' folders)
            split: 'train', 'val', or 'test'
            transform: Image transforms (deprecated, use augmentation)
            mask_transform: Mask transforms (deprecated, use augmentation)
            use_subject_split: Whether to use subject-aware splitting
            preserve_aspect: Whether to preserve aspect ratio
            image_size: Target image size
            seed: Random seed for reproducible splits
        """
        self.dataset_root = dataset_root
        self.images_dir = os.path.join(dataset_root, 'images')
        self.masks_dir = os.path.join(dataset_root, 'masks')
        self.split = split
        self.use_subject_split = use_subject_split
        self.preserve_aspect = preserve_aspect
        self.image_size = image_size
        self.seed = seed
        
        # Verify directories exist
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not os.path.exists(self.masks_dir):
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")
        
        # Legacy transform support (deprecated)
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Use new augmentation pipeline
        self.augmentation = IrisAugmentation(
            image_size=image_size,
            training=(split == 'train'),
            preserve_aspect=preserve_aspect
        )
        
        # FIXED: Get all image-mask pairs (supporting multiple formats)
        all_pairs = self._load_image_mask_pairs()
        
        # Split dataset
        if self.use_subject_split:
            self.image_files, self.mask_files = self._subject_aware_split(all_pairs, split)
        else:
            self.image_files, self.mask_files = self._random_split(all_pairs, split)
        
        print(f"✅ {split.upper()} dataset: {len(self.image_files)} samples")
        
        # Default transforms if augmentation not available
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        if self.mask_transform is None:
            self.mask_transform = transforms.Compose([
                transforms.Resize((image_size, image_size), 
                                interpolation=transforms.InterpolationMode.NEAREST)
            ])
    
    def _load_image_mask_pairs(self):
        """
        FIXED: Load image-mask pairs with support for multiple formats
        """
        all_pairs = []
        
        print(f"📂 Scanning directories...")
        print(f"   Images: {self.images_dir}")
        print(f"   Masks: {self.masks_dir}")
        
        for img_file in os.listdir(self.images_dir):
            file_ext = os.path.splitext(img_file.lower())[1]
            
            # Check if valid image extension
            if file_ext not in IMAGE_EXTENSIONS:
                continue
            
            # Extract base name (without extension)
            base_name = os.path.splitext(img_file)[0]
            
            # Find corresponding mask (try different extensions and patterns)
            mask_file = None
            mask_path = None
            
            # Try different mask naming patterns
            mask_patterns = [
                f"OperatorA_{base_name}",  # Standard UBIRIS pattern
                base_name,                  # Same name as image
                f"{base_name}_mask",       # Alternative pattern
                f"mask_{base_name}"        # Another alternative
            ]
            
            for pattern in mask_patterns:
                for mask_ext in IMAGE_EXTENSIONS:
                    potential_mask = f"{pattern}{mask_ext}"
                    potential_path = os.path.join(self.masks_dir, potential_mask)
                    
                    if os.path.exists(potential_path):
                        mask_file = potential_mask
                        mask_path = potential_path
                        break
                
                if mask_file:
                    break
            
            if mask_file is None:
                # Warn about missing mask but continue
                continue
            
            # Extract subject ID for subject-aware split
            # UBIRIS format: C{camera}_S{session}_I{image}.ext
            camera_match = re.search(r'C(\d+)', base_name)
            session_match = re.search(r'S(\d+)', base_name)
            
            if camera_match and session_match:
                camera_id = int(camera_match.group(1))
                session_id = int(session_match.group(1))
                subject_id = camera_id * 1000 + session_id
            else:
                subject_id = 0
            
            all_pairs.append({
                'image_file': img_file,
                'mask_file': mask_file,
                'subject_id': subject_id
            })
        
        if len(all_pairs) == 0:
            raise RuntimeError(
                f"No valid image-mask pairs found!\n"
                f"Images dir: {self.images_dir}\n"
                f"Masks dir: {self.masks_dir}\n"
                f"Supported formats: {IMAGE_EXTENSIONS}\n"
                f"Expected mask patterns: OperatorA_[basename].ext or [basename].ext"
            )
        
        print(f"📊 Found {len(all_pairs)} image-mask pairs")
        
        # Print format distribution
        formats = {}
        for pair in all_pairs:
            ext = os.path.splitext(pair['image_file'].lower())[1]
            formats[ext] = formats.get(ext, 0) + 1
        
        print(f"📁 Image formats:")
        for ext, count in formats.items():
            print(f"   {ext}: {count} files")
        
        return all_pairs
    
    def _subject_aware_split(self, all_pairs, split):
        """
        Split dataset by subjects to prevent data leakage
        """
        # Group by subject
        subjects = {}
        for pair in all_pairs:
            subject_id = pair['subject_id']
            if subject_id not in subjects:
                subjects[subject_id] = []
            subjects[subject_id].append(pair)
        
        # Split subjects (80% train, 10% val, 10% test)
        subject_ids = list(subjects.keys())
        subject_ids.sort()  # Ensure reproducibility
        
        train_subjects, temp_subjects = train_test_split(
            subject_ids, test_size=0.2, random_state=self.seed
        )
        val_subjects, test_subjects = train_test_split(
            temp_subjects, test_size=0.5, random_state=self.seed
        )
        
        # Get files for this split
        if split == 'train':
            split_subjects = train_subjects
        elif split == 'val':
            split_subjects = val_subjects
        elif split == 'test':
            split_subjects = test_subjects
        else:
            raise ValueError(f"Unknown split: {split}")
        
        image_files = []
        mask_files = []
        for subject_id in split_subjects:
            for pair in subjects[subject_id]:
                image_files.append(pair['image_file'])
                mask_files.append(pair['mask_file'])
        
        print(f"   Subject-aware split: {len(split_subjects)} subjects")
        
        return image_files, mask_files
    
    def _random_split(self, all_pairs, split):
        """
        Random split (fallback if subject-aware split not used)
        """
        image_files = [pair['image_file'] for pair in all_pairs]
        mask_files = [pair['mask_file'] for pair in all_pairs]
        
        # Random split (80% train, 10% val, 10% test)
        total_samples = len(image_files)
        train_end = int(0.8 * total_samples)
        val_end = int(0.9 * total_samples)
        
        if split == 'train':
            return image_files[:train_end], mask_files[:train_end]
        elif split == 'val':
            return image_files[train_end:val_end], mask_files[train_end:val_end]
        elif split == 'test':
            return image_files[val_end:], mask_files[val_end:]
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Returns:
            dict with keys:
                - pixel_values: [3, H, W] normalized image tensor
                - labels: [H, W] segmentation mask (0=background/pupil, 1=iris)
                - boundary: [H, W] boundary mask
                - image_path: str
                - mask_path: str
        """
        # Load image and mask
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        
        try:
            # FIXED: Open with PIL (supports TIFF)
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')  # Grayscale
        except Exception as e:
            raise RuntimeError(
                f"Failed to load sample {idx}:\n"
                f"  Image: {img_path}\n"
                f"  Mask: {mask_path}\n"
                f"  Error: {e}"
            )
        
        # Convert mask to numpy for preprocessing
        mask_np = np.array(mask)
        
        # Preprocess mask for iris segmentation:
        # Threshold to binary: 0 = background/pupil, 1 = iris
        processed_mask = np.zeros_like(mask_np, dtype=np.uint8)
        processed_mask[mask_np > 127] = 1  # Threshold at 127
        
        # Use new augmentation pipeline if available
        if hasattr(self, 'augmentation') and self.augmentation is not None:
            try:
                image_tensor, mask_tensor, boundary_tensor = self.augmentation(
                    image, processed_mask
                )
                
                return {
                    'pixel_values': image_tensor,
                    'labels': mask_tensor,
                    'boundary': boundary_tensor,
                    'image_path': img_path,
                    'mask_path': mask_path
                }
            except Exception as e:
                print(f"⚠️  Augmentation failed for {img_path}: {e}")
                # Fall through to legacy transforms
        
        # Fallback to legacy transforms
        processed_mask_pil = Image.fromarray(processed_mask.astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            processed_mask_pil = self.mask_transform(processed_mask_pil)
        
        # Convert mask to tensor manually
        mask_tensor = torch.from_numpy(np.array(processed_mask_pil)).long()
        
        # Create boundary tensor for compatibility
        boundary_tensor = torch.from_numpy(
            create_boundary_mask(np.array(processed_mask_pil))
        ).float()
        
        return {
            'pixel_values': image,
            'labels': mask_tensor,
            'boundary': boundary_tensor,
            'image_path': img_path,
            'mask_path': mask_path
        }


# Test function
if __name__ == "__main__":
    print("Testing UbirisDataset with TIFF support...")
    
    try:
        # Test loading
        ds = UbirisDataset('dataset', split='train')
        
        print(f"\n✅ Dataset loaded: {len(ds)} samples")
        
        # Test getting a sample
        sample = ds[0]
        
        print(f"✅ Sample loaded:")
        print(f"   Image shape: {sample['pixel_values'].shape}")
        print(f"   Mask shape: {sample['labels'].shape}")
        print(f"   Boundary shape: {sample['boundary'].shape}")
        print(f"   Mask values: {torch.unique(sample['labels'])}")
        
        print("\n✅ Dataset test passed!")
        
    except Exception as e:
        print(f"\n❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
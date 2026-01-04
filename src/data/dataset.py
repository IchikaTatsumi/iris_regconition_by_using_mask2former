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
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from transforms import IrisAugmentation, create_boundary_mask


class UbirisDataset(Dataset):
    def __init__(self, dataset_root, split='train', transform=None, mask_transform=None, 
                 use_subject_split=True, preserve_aspect=True, image_size=512, seed=42):
        """
        UBIRIS V2 Dataset for iris and pupil segmentation
        
        Args:
            dataset_root: Path to dataset root directory (should contain 'images' and 'masks' folders)
            split: 'train', 'val', or 'test'
            transform: Transformations to apply to input images (deprecated, use augmentation)
            mask_transform: Transformations to apply to mask images (deprecated, use augmentation)
            use_subject_split: Whether to use subject-aware splitting
            preserve_aspect: Whether to preserve aspect ratio during resize
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
        
        # Legacy transform support (deprecated)
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Use new augmentation pipeline
        self.augmentation = IrisAugmentation(
            image_size=image_size,
            training=(split == 'train'),
            preserve_aspect=preserve_aspect
        )
        
        # Get all image-mask pairs
        all_pairs = []
        for img_file in os.listdir(self.images_dir):
            if img_file.endswith('.png'):
                # Extract identifiers from image filename (C{class}_S{session}_I{image}.png)
                base_name = img_file.replace('.png', '')
                # Find corresponding mask file (OperatorA_{base_name}.png)
                mask_file = f"OperatorA_{base_name}.png"
                mask_path = os.path.join(self.masks_dir, mask_file)
                
                if os.path.exists(mask_path):
                    # Extract actual subject ID from filename pattern
                    # UBIRIS format: C{camera}_S{session}_I{image}.png
                    # For true subject-aware split, we need session info
                    # Subject = Camera + Session combination for person independence
                    camera_match = re.search(r'C(\d+)_', img_file)
                    session_match = re.search(r'S(\d+)_', img_file)
                    
                    if camera_match and session_match:
                        camera_id = int(camera_match.group(1))
                        session_id = int(session_match.group(1))
                        # Create unique subject ID: camera*1000 + session
                        # This ensures different people don't share IDs
                        subject_id = camera_id * 1000 + session_id
                    else:
                        subject_id = 0
                    
                    all_pairs.append({
                        'image_file': img_file,
                        'mask_file': mask_file,
                        'subject_id': subject_id
                    })
        
        # Split dataset with subject awareness
        if self.use_subject_split:
            self.image_files, self.mask_files = self._subject_aware_split(all_pairs, split)
        else:
            # Original random split
            self.image_files, self.mask_files = self._random_split(all_pairs, split)
        
        # Default transforms if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((300, 400)),  # Keep original size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        if self.mask_transform is None:
            self.mask_transform = transforms.Compose([
                transforms.Resize((300, 400), interpolation=Image.NEAREST)
            ])
    
    def _subject_aware_split(self, all_pairs, split):
        """Split dataset by subjects to prevent data leakage"""
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
        
        print(f"Subject-aware split - {split}: {len(split_subjects)} subjects, {len(image_files)} samples")
        return image_files, mask_files
    
    def _random_split(self, all_pairs, split):
        """Random split (original method)"""
        # Extract files
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
        # Load image and mask
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale
        
        # Convert mask to numpy for preprocessing
        mask_np = np.array(mask)
        
        # Preprocess mask for iris segmentation:
        # Original: 255 = iris, 0 = background + pupil
        # SegFormer expects: 0 = background/pupil, 1 = iris
        
        # Debug: Check for anti-aliasing issues
        unique_vals = np.unique(mask_np)
        if len(unique_vals) > 2:
            print(f"⚠️  Mask has {len(unique_vals)} unique values: {unique_vals}")
            print(f"   File: {mask_path}")
        
        # Fix anti-aliasing: use threshold instead of exact match
        processed_mask = np.zeros_like(mask_np, dtype=np.uint8)
        processed_mask[mask_np > 127] = 1  # Use threshold to handle anti-aliased pixels
        
        # Use new augmentation pipeline if available
        if hasattr(self, 'augmentation') and self.augmentation is not None:
            try:
                image_tensor, mask_tensor, boundary_tensor = self.augmentation(image, processed_mask)
                
                return {
                    'pixel_values': image_tensor,
                    'labels': mask_tensor,
                    'boundary': boundary_tensor,
                    'image_path': img_path,
                    'mask_path': mask_path
                }
            except Exception as e:
                print(f"⚠️  Augmentation failed, falling back to legacy transforms: {e}")
                print(f"   File: {img_path}")
                # Important: The fallback path may cause double normalization!
        
        # Fallback to legacy transforms
        processed_mask_pil = Image.fromarray(processed_mask.astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            processed_mask_pil = self.mask_transform(processed_mask_pil)
        
        # Convert mask to tensor manually to avoid normalization
        mask_tensor = torch.from_numpy(np.array(processed_mask_pil)).long()
        
        # Create boundary tensor for compatibility
        boundary_tensor = torch.from_numpy(create_boundary_mask(np.array(processed_mask_pil))).float()
        
        return {
            'pixel_values': image,
            'labels': mask_tensor,
            'boundary': boundary_tensor,
            'image_path': img_path,
            'mask_path': mask_path
        }

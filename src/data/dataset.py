"""
MEMORY-SAFE dataset.py - FIXED for Kaggle 16GB RAM
Key fixes:
1. Lazy loading - chỉ load khi __getitem__
2. Clear transforms sau mỗi sample
3. Giảm memory footprint
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import gc

from .transforms import IrisAugmentation, create_boundary_mask

IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']


class UbirisDataset(Dataset):
    """
    MEMORY-SAFE UBIRIS V2 Dataset - Kaggle optimized
    """
    
    def __init__(
        self, 
        dataset_root, 
        split='train', 
        use_subject_split=True, 
        preserve_aspect=True, 
        image_size=512, 
        seed=42
    ):
        self.dataset_root = dataset_root
        self.images_dir = os.path.join(dataset_root, 'images')
        self.masks_dir = os.path.join(dataset_root, 'masks')
        self.split = split
        self.use_subject_split = use_subject_split
        self.preserve_aspect = preserve_aspect
        self.image_size = image_size
        self.seed = seed
        
        # Verify directories
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not os.path.exists(self.masks_dir):
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")
        
        # ✅ Augmentation pipeline
        self.augmentation = IrisAugmentation(
            image_size=image_size,
            training=(split == 'train'),
            preserve_aspect=preserve_aspect
        )
        
        # ✅ Load ONLY file paths (không load ảnh)
        all_pairs = self._load_image_mask_pairs()
        
        # Split dataset
        if self.use_subject_split:
            self.image_files, self.mask_files = self._subject_aware_split(all_pairs, split)
        else:
            self.image_files, self.mask_files = self._random_split(all_pairs, split)
        
        print(f"✅ {split.upper()} dataset: {len(self.image_files)} samples")
        
        # 🔥 KAGGLE FIX: Thêm flag để force garbage collection
        self._gc_counter = 0
        self._gc_frequency = 100  # GC mỗi 100 samples
    
    def _load_image_mask_pairs(self):
        """
        ✅ Load ONLY file paths - KHÔNG load ảnh vào RAM
        """
        all_pairs = []
        
        print(f"📂 Scanning directories...")
        
        for img_file in os.listdir(self.images_dir):
            file_ext = os.path.splitext(img_file.lower())[1]
            
            if file_ext not in IMAGE_EXTENSIONS:
                continue
            
            base_name = os.path.splitext(img_file)[0]
            
            # Tìm mask tương ứng
            mask_file = None
            mask_patterns = [
                f"OperatorA_{base_name}",
                base_name,
                f"{base_name}_mask",
                f"mask_{base_name}"
            ]
            
            for pattern in mask_patterns:
                for mask_ext in IMAGE_EXTENSIONS:
                    potential_mask = f"{pattern}{mask_ext}"
                    potential_path = os.path.join(self.masks_dir, potential_mask)
                    
                    if os.path.exists(potential_path):
                        mask_file = potential_mask
                        break
                
                if mask_file:
                    break
            
            if mask_file is None:
                continue
            
            # Extract subject ID
            import re
            camera_match = re.search(r'C(\d+)', base_name)
            session_match = re.search(r'S(\d+)', base_name)
            
            if camera_match and session_match:
                camera_id = int(camera_match.group(1))
                session_id = int(session_match.group(1))
                subject_id = camera_id * 1000 + session_id
            else:
                subject_id = 0
            
            # ✅ Chỉ lưu TÊN FILE, không load ảnh
            all_pairs.append({
                'image_file': img_file,
                'mask_file': mask_file,
                'subject_id': subject_id
            })
        
        if len(all_pairs) == 0:
            raise RuntimeError(
                f"No valid image-mask pairs found!\n"
                f"Images dir: {self.images_dir}\n"
                f"Masks dir: {self.masks_dir}"
            )
        
        print(f"📊 Found {len(all_pairs)} image-mask pairs")
        
        return all_pairs
    
    def _subject_aware_split(self, all_pairs, split):
        """Split by subjects"""
        subjects = {}
        for pair in all_pairs:
            subject_id = pair['subject_id']
            if subject_id not in subjects:
                subjects[subject_id] = []
            subjects[subject_id].append(pair)
        
        subject_ids = list(subjects.keys())
        subject_ids.sort()
        
        train_subjects, temp_subjects = train_test_split(
            subject_ids, test_size=0.2, random_state=self.seed
        )
        val_subjects, test_subjects = train_test_split(
            temp_subjects, test_size=0.5, random_state=self.seed
        )
        
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
        
        return image_files, mask_files
    
    def _random_split(self, all_pairs, split):
        """Random split"""
        image_files = [pair['image_file'] for pair in all_pairs]
        mask_files = [pair['mask_file'] for pair in all_pairs]
        
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
        🔥 LAZY LOADING - Chỉ load khi cần
        """
        # Build full paths
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        
        try:
            # 🔥 LOAD ẢNH Ở ĐÂY - không phải ở __init__
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
        except Exception as e:
            raise RuntimeError(
                f"Failed to load sample {idx}:\n"
                f"  Image: {img_path}\n"
                f"  Mask: {mask_path}\n"
                f"  Error: {e}"
            )
        
        # Convert mask to numpy
        mask_np = np.array(mask)
        
        # Threshold mask: 0=background/pupil, 1=iris
        processed_mask = np.zeros_like(mask_np, dtype=np.uint8)
        processed_mask[mask_np > 127] = 1
        
        # 🔥 Apply augmentation
        try:
            image_tensor, mask_tensor, boundary_tensor = self.augmentation(
                image, processed_mask
            )
        except Exception as e:
            print(f"⚠️ Augmentation failed for {img_path}: {e}")
            # Fallback: simple conversion
            from torchvision import transforms
            image_tensor = transforms.ToTensor()(image)
            mask_tensor = torch.from_numpy(processed_mask).long()
            boundary_tensor = torch.from_numpy(
                create_boundary_mask(processed_mask)
            ).float()
        
        # 🔥 KAGGLE FIX: Force garbage collection periodically
        self._gc_counter += 1
        if self._gc_counter >= self._gc_frequency:
            gc.collect()
            self._gc_counter = 0
        
        return {
            'pixel_values': image_tensor,
            'labels': mask_tensor,
            'boundary': boundary_tensor,
            'image_path': img_path,
            'mask_path': mask_path
        }


if __name__ == "__main__":
    print("Testing MEMORY-SAFE UbirisDataset...")
    
    try:
        ds = UbirisDataset('dataset', split='train')
        
        print(f"\n✅ Dataset loaded: {len(ds)} samples")
        
        # Test getting a sample
        sample = ds[0]
        
        print(f"✅ Sample loaded:")
        print(f"   Image shape: {sample['pixel_values'].shape}")
        print(f"   Mask shape: {sample['labels'].shape}")
        print(f"   Boundary shape: {sample['boundary'].shape}")
        
        print("\n✅ Dataset test passed!")
        
    except Exception as e:
        print(f"\n❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
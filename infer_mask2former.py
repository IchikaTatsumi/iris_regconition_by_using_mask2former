#!/usr/bin/env python3
"""
FIXED Command line interface for Mask2Former iris segmentation inference
"""

import sys
import os
import argparse
import torch
import json
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import model class trực tiếp thay vì qua lớp trung gian để dễ kiểm soát
try:
    from src.models.mask2former import EnhancedMask2Former
except ImportError:
    print("❌ LỖI: Không tìm thấy module 'src'.")
    sys.exit(1)

class Mask2FormerInference:
    def __init__(self, checkpoint_path, config_path, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Device: {self.device}")

        # 1. Load Config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        model_cfg = self.config.get('model', self.config.get('model_config', {}))
        # Lọc bỏ key thừa
        for k in ['architecture', 'model_type', 'use_checkpoint']:
            if k in model_cfg: del model_cfg[k]

        # 2. Init Model
        try:
            self.model = EnhancedMask2Former(**model_cfg)
        except TypeError:
            # Fallback tối giản
            minimal_cfg = {k:v for k,v in model_cfg.items() if k in ['num_labels', 'model_name', 'num_queries']}
            self.model = EnhancedMask2Former(**minimal_cfg)

        # 3. Load Weights (Fix lỗi PyTorch 2.6)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt.get('model', ckpt)))
            self.model.load_state_dict(state_dict)
            print("✅ Weights loaded successfully")
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            sys.exit(1)

        self.model.to(self.device).eval()

        # 4. Transform (Chuẩn 320x320)
        self.img_size = 320
        self.transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def predict(self, image_path, confidence_threshold=0.35):
        # Đọc ảnh
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        original_h, original_w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess
        x = self.transform(image=image_rgb)['image'].unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                out = self.model(x)
                logits = out.get('pred_masks', out.get('logits', out))
                
                # Softmax & Threshold
                probs = F.softmax(logits, dim=1)
                iris_prob = probs[0, 1, :, :] # Class 1
                
                # Tạo mask nhị phân
                mask_small = (iris_prob > confidence_threshold).float()

        # Post-process (Resize mask về kích thước gốc)
        mask_np = mask_small.cpu().numpy().astype(np.uint8)
        mask_resized = cv2.resize(mask_np, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        
        return {
            'mask': mask_resized,
            'confidence': float(iris_prob.max())
        }

    def save_prediction(self, image_path, result, output_path, overlay=True):
        image = cv2.imread(str(image_path))
        mask = result['mask']
        
        # Tạo overlay
        if overlay:
            colored_mask = np.zeros_like(image)
            colored_mask[mask == 1] = [0, 255, 0] # Green
            image = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
            
            # Vẽ contour
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, cnts, -1, (0, 255, 255), 2)

        cv2.imwrite(str(output_path), image)


def main():
    parser = argparse.ArgumentParser(description='Mask2Former Iris Segmentation Inference')
    
    # Input options
    parser.add_argument('--image', type=str, help='Input image path')
    parser.add_argument('--batch', type=str, help='Input directory for batch processing')
    
    # Model options
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_checkpoint.pth', help='Path to checkpoint')
    parser.add_argument('--config', type=str, default='configs/mask2former_config_kaggle.json', help='Path to config')
    
    # Output options
    parser.add_argument('--output', type=str, default='inference_results', help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.35, help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Validate
    if not args.image and not args.batch:
        print("❌ Error: Must specify either --image or --batch")
        return

    # Init Inference Engine
    try:
        engine = Mask2FormerInference(args.checkpoint, args.config)
    except Exception as e:
        print(f"❌ Init Error: {e}")
        return

    # Create output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Single Image Mode
    if args.image:
        print(f"📸 Processing: {args.image}")
        try:
            res = engine.predict(args.image, args.threshold)
            out_path = output_dir / Path(args.image).name
            engine.save_prediction(args.image, res, out_path)
            print(f"✅ Saved to: {out_path} (Conf: {res['confidence']:.2f})")
        except Exception as e:
            print(f"❌ Error: {e}")

    # 2. Batch Mode
    elif args.batch:
        input_dir = Path(args.batch)
        extensions = ['*.jpg', '*.png', '*.jpeg']
        files = []
        for ext in extensions:
            files.extend(input_dir.glob(ext))
            
        print(f"🚀 Processing batch: {len(files)} images")
        
        for idx, img_path in enumerate(files):
            try:
                res = engine.predict(img_path, args.threshold)
                out_path = output_dir / img_path.name
                engine.save_prediction(img_path, res, out_path)
                print(f"[{idx+1}/{len(files)}] ✅ {img_path.name} (Conf: {res['confidence']:.2f})")
            except Exception as e:
                print(f"[{idx+1}/{len(files)}] ❌ {img_path.name}: {e}")

if __name__ == "__main__":
    main()
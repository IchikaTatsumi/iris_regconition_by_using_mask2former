#!/usr/bin/env python3
"""
Test script for overlay visualization functionality (Fixed)
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# Import class inference từ file infer_mask2former.py (đảm bảo file đó nằm cùng thư mục)
try:
    from infer_mask2former import Mask2FormerInference
except ImportError:
    print("❌ LỖI: Không tìm thấy file 'infer_mask2former.py'.")
    print("👉 Hãy đảm bảo file test_overlay.py nằm cùng thư mục với infer_mask2former.py")
    sys.exit(1)

def create_overlay_visualization(image_path, result, color=(0, 255, 0), alpha=0.5):
    """Hàm vẽ overlay đơn giản để test"""
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = result['mask']
    
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 1] = color
    
    overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
    return overlay

def test_overlay_visualization():
    # Cấu hình đường dẫn (Sửa lại cho đúng file của bạn)
    checkpoint_path = "checkpoints/best_checkpoint.pth"
    config_path = "configs/mask2former_config_kaggle.json"
    
    # Tạo một ảnh mẫu giả lập nếu không có file thật (hoặc thay bằng đường dẫn ảnh thật)
    sample_image = "test_eye.jpg" 
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    if not os.path.exists(sample_image):
        print(f"❌ Sample image not found: {sample_image}")
        print("👉 Hãy copy một ảnh mắt vào và đổi tên thành 'eye_test.jpg'")
        return
    
    print("🎨 Testing overlay visualization functionality")
    
    try:
        # Load model
        print("⏳ Loading model...")
        model = Mask2FormerInference(checkpoint_path, config_path)
        
        # Run inference
        print("🚀 Running inference...")
        results = model.predict(sample_image)
        
        # Create output directory
        output_dir = Path("overlay_test_results")
        output_dir.mkdir(exist_ok=True)
        
        print("🎨 Creating overlay visualizations...")
        
        # Test 1: Basic Green Overlay
        print("   1️⃣ Green Overlay")
        overlay1 = create_overlay_visualization(sample_image, results, color=(0, 255, 0))
        Image.fromarray(overlay1).save(output_dir / "overlay_green.png")
        
        # Test 2: Red Overlay (Iris only style)
        print("   2️⃣ Red Overlay")
        overlay2 = create_overlay_visualization(sample_image, results, color=(255, 0, 0), alpha=0.3)
        Image.fromarray(overlay2).save(output_dir / "overlay_red.png")
        
        # Test 3: Blue Overlay
        print("   3️⃣ Blue Overlay")
        overlay3 = create_overlay_visualization(sample_image, results, color=(0, 0, 255), alpha=0.6)
        Image.fromarray(overlay3).save(output_dir / "overlay_blue.png")

        print(f"✅ All overlay tests completed!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📊 Confidence: {results['confidence']:.3f}")

    except Exception as e:
        print(f"❌ Error during overlay testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_overlay_visualization()
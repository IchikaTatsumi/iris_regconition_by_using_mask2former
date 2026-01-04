#!/usr/bin/env python3
"""
Test script for overlay visualization functionality
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from inference import IrisSegmentationInference
from utils.visualization import create_overlay_visualization, create_comparison_visualization


def test_overlay_visualization():
    """Test overlay visualization with different settings"""
    
    checkpoint_path = "outputs/segformer_iris_a100/checkpoints/best.pt"
    sample_image = "dataset/images/C100_S1_I10.png"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    if not os.path.exists(sample_image):
        print(f"❌ Sample image not found: {sample_image}")
        return
    
    print("🎨 Testing overlay visualization functionality")
    print(f"📋 Model: {checkpoint_path}")
    print(f"📸 Image: {sample_image}")
    
    try:
        # Load model
        print("⏳ Loading model...")
        model = IrisSegmentationInference(checkpoint_path)
        
        # Run inference
        print("🚀 Running inference...")
        results = model.predict(sample_image)
        
        # Create output directory
        output_dir = Path("overlay_test_results")
        output_dir.mkdir(exist_ok=True)
        
        print("🎨 Creating overlay visualizations...")
        
        # Test 1: Basic overlay with default settings
        print("   1️⃣ Basic overlay (red iris, cyan boundary)")
        overlay_basic = create_overlay_visualization(sample_image, results)
        from PIL import Image
        Image.fromarray(overlay_basic).save(output_dir / "overlay_basic.png")
        
        # Test 2: Custom colors and transparency
        print("   2️⃣ Custom overlay (green iris, yellow boundary)")
        overlay_custom = create_overlay_visualization(
            sample_image, 
            results,
            iris_color=(0, 255, 0),      # Green
            boundary_color=(255, 255, 0), # Yellow
            iris_alpha=0.6,
            boundary_alpha=0.9,
            boundary_thickness=3
        )
        Image.fromarray(overlay_custom).save(output_dir / "overlay_custom.png")
        
        # Test 3: Iris only (no boundary)
        print("   3️⃣ Iris only overlay (purple iris)")
        overlay_iris_only = create_overlay_visualization(
            sample_image, 
            results,
            iris_color=(128, 0, 128),     # Purple
            show_boundary=False,
            iris_alpha=0.5
        )
        Image.fromarray(overlay_iris_only).save(output_dir / "overlay_iris_only.png")
        
        # Test 4: Side-by-side comparison
        print("   4️⃣ Comparison visualization")
        create_comparison_visualization(
            sample_image, 
            results, 
            output_dir / "comparison.png"
        )
        
        # Test 5: Using inference class method
        print("   5️⃣ Using inference class overlay method")
        overlay_class = model.create_overlay(
            sample_image,
            results,
            iris_color=(255, 0, 255),     # Magenta
            boundary_color=(0, 255, 0),   # Green
            iris_alpha=0.3,
            boundary_alpha=0.7
        )
        Image.fromarray(overlay_class).save(output_dir / "overlay_class_method.png")
        
        # Test 6: Full save_prediction with overlays
        print("   6️⃣ Full prediction save with overlays")
        model.save_prediction(
            results,
            output_dir / "full_prediction",
            original_image=sample_image,
            save_overlay=True,
            save_comparison=True,
            save_components=True,
            overlay_kwargs={
                'iris_color': (0, 0, 255),    # Blue
                'boundary_color': (255, 165, 0), # Orange
                'iris_alpha': 0.45,
                'boundary_alpha': 0.85
            }
        )
        
        print(f"✅ All overlay tests completed!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"🔍 Check the following files:")
        print(f"   - overlay_basic.png (default red/cyan)")
        print(f"   - overlay_custom.png (green/yellow)")
        print(f"   - overlay_iris_only.png (purple iris only)")
        print(f"   - comparison.png (side-by-side comparison)")
        print(f"   - overlay_class_method.png (using class method)")
        print(f"   - full_prediction_*.png (complete prediction set)")
        
        # Print statistics
        seg_mask = results['segmentation']['mask']
        iris_coverage = (seg_mask == 1).sum() / seg_mask.size * 100
        avg_confidence = results['segmentation']['confidence'].mean()
        
        print(f"\n📊 Image Statistics:")
        print(f"   - Image size: {seg_mask.shape}")
        print(f"   - Iris coverage: {iris_coverage:.1f}%")
        print(f"   - Average confidence: {avg_confidence:.3f}")
        
        if 'boundary' in results:
            boundary_density = results['boundary']['boundary_mask'].sum() / results['boundary']['boundary_mask'].size * 100
            print(f"   - Boundary density: {boundary_density:.1f}%")
        
    except Exception as e:
        print(f"❌ Error during overlay testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_overlay_visualization()

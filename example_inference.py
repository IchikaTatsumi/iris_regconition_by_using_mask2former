#!/usr/bin/env python3
"""
FIXED Example script showing how to use the iris segmentation inference module
Key fixes:
1. Import Mask2FormerInference (correct class name)
2. Support both Mask2Former and SegFormer checkpoints
3. Better error handling and user feedback
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.inference import Mask2FormerInference  # FIXED: Correct class name
import torch


def example_single_inference():
    """FIXED: Example of running inference on a single image"""
    
    print("="*60)
    print("SINGLE IMAGE INFERENCE EXAMPLE")
    print("="*60)
    
    # Try to find available checkpoints
    possible_checkpoints = [
        "outputs/mask2former_iris/checkpoints/best.pt",  # Mask2Former
        "outputs/segformer_iris_a100/checkpoints/best.pt",  # SegFormer (fallback)
        "outputs/segformer_iris/checkpoints/best.pt",  # SegFormer alternative
    ]
    
    checkpoint_path = None
    for path in possible_checkpoints:
        if os.path.exists(path):
            checkpoint_path = path
            print(f"✅ Found checkpoint: {checkpoint_path}")
            break
    
    if not checkpoint_path:
        print(f"❌ No checkpoint found. Tried:")
        for path in possible_checkpoints:
            print(f"   - {path}")
        print("\nPlease ensure you have trained the model first.")
        return
    
    # Find sample images
    sample_images = []
    if Path("dataset/images").exists():
        sample_images = list(Path("dataset/images").glob("*.png"))[:3]
    
    if not sample_images:
        print("❌ No sample images found in dataset/images/")
        print("Please ensure your dataset is available.")
        return
    
    print(f"📸 Found {len(sample_images)} sample images")
    print("="*60 + "\n")
    
    # Load the inference model
    print("⏳ Loading model...")
    try:
        model = Mask2FormerInference(checkpoint_path)
        print("✅ Model loaded successfully\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Process each sample image
    for i, image_path in enumerate(sample_images):
        print(f"📸 Processing {i+1}/{len(sample_images)}: {image_path.name}")
        
        try:
            # Run inference
            results = model.predict(image_path)
            
            # Print statistics
            seg_mask = results['segmentation']['mask']
            iris_coverage = (seg_mask == 1).sum() / seg_mask.size * 100
            avg_confidence = results['segmentation']['confidence'].mean()
            
            print(f"   📊 Iris coverage: {iris_coverage:.1f}%")
            print(f"   🎯 Average confidence: {avg_confidence:.3f}")
            
            if 'boundary' in results:
                boundary_density = results['boundary']['boundary_mask'].sum() / results['boundary']['boundary_mask'].size * 100
                print(f"   🔲 Boundary density: {boundary_density:.1f}%")
            
            # Save results
            output_dir = Path("inference_results")
            output_dir.mkdir(exist_ok=True)
            
            model.save_prediction(
                results,
                output_dir / f"result_{i+1}_{image_path.stem}",
                original_image=image_path,
                save_components=True,
                save_overlay=True,
                save_comparison=True
            )
            print(f"   ✅ Saved results\n")
            
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
    
    print("="*60)
    print("✅ Single image inference completed!")
    print(f"📁 Results saved to: inference_results/")
    print("="*60 + "\n")


def example_batch_inference():
    """FIXED: Example of running batch inference"""
    
    print("="*60)
    print("BATCH INFERENCE EXAMPLE")
    print("="*60)
    
    # Try to find checkpoint
    possible_checkpoints = [
        "outputs/mask2former_iris/checkpoints/best.pt",
        "outputs/segformer_iris_a100/checkpoints/best.pt",
        "outputs/segformer_iris/checkpoints/best.pt",
    ]
    
    checkpoint_path = None
    for path in possible_checkpoints:
        if os.path.exists(path):
            checkpoint_path = path
            print(f"✅ Found checkpoint: {checkpoint_path}")
            break
    
    if not checkpoint_path:
        print(f"❌ No checkpoint found")
        return
    
    # Get multiple images
    sample_images = []
    if Path("dataset/images").exists():
        sample_images = list(Path("dataset/images").glob("*.png"))[:5]
    
    if len(sample_images) < 2:
        print("❌ Need at least 2 images for batch inference")
        return
    
    print(f"📸 Processing {len(sample_images)} images")
    print("="*60 + "\n")
    
    # Load model
    print("⏳ Loading model...")
    try:
        model = Mask2FormerInference(checkpoint_path)
        print("✅ Model loaded\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Run batch inference
    print("🚀 Running batch inference...")
    try:
        results_list = model.predict_batch(sample_images)
    except Exception as e:
        print(f"❌ Batch inference failed: {e}")
        return
    
    # Process results
    print("\n📊 Results:")
    for i, (image_path, results) in enumerate(zip(sample_images, results_list), 1):
        seg_mask = results['segmentation']['mask']
        iris_coverage = (seg_mask == 1).sum() / seg_mask.size * 100
        print(f"   {i}. {image_path.name}: {iris_coverage:.1f}% iris coverage")
    
    print("\n✅ Batch inference completed!")
    print("="*60 + "\n")


def example_quick_inference():
    """FIXED: Example using quick inference"""
    
    print("="*60)
    print("QUICK INFERENCE EXAMPLE")
    print("="*60)
    
    # Try to find checkpoint
    possible_checkpoints = [
        "outputs/mask2former_iris/checkpoints/best.pt",
        "outputs/segformer_iris_a100/checkpoints/best.pt",
        "outputs/segformer_iris/checkpoints/best.pt",
    ]
    
    checkpoint_path = None
    for path in possible_checkpoints:
        if os.path.exists(path):
            checkpoint_path = path
            print(f"✅ Found checkpoint: {checkpoint_path}")
            break
    
    if not checkpoint_path:
        print(f"❌ No checkpoint found")
        return
    
    # Find one sample image
    sample_images = []
    if Path("dataset/images").exists():
        sample_images = list(Path("dataset/images").glob("*.png"))[:1]
    
    if not sample_images:
        print("❌ No sample images found")
        return
    
    image_path = sample_images[0]
    print(f"📸 Image: {image_path.name}")
    print("="*60 + "\n")
    
    print("🚀 Running quick inference...")
    
    try:
        # Load model and predict
        model = Mask2FormerInference(checkpoint_path)
        results = model.predict(image_path)
        
        # Save with visualization
        output_dir = Path("quick_inference_results")
        output_dir.mkdir(exist_ok=True)
        
        model.save_prediction(
            results,
            output_dir / "quick_result",
            original_image=image_path,
            save_overlay=True,
            save_comparison=True,
            save_components=True
        )
        
        seg_mask = results['segmentation']['mask']
        iris_coverage = (seg_mask == 1).sum() / seg_mask.size * 100
        
        print(f"✅ Quick inference completed!")
        print(f"📊 Iris coverage: {iris_coverage:.1f}%")
        print(f"📁 Results saved to: quick_inference_results/")
        
    except Exception as e:
        print(f"❌ Quick inference failed: {e}")
    
    print("="*60 + "\n")


def main():
    """Main function with menu"""
    
    print("\n" + "="*60)
    print("🔬 IRIS SEGMENTATION INFERENCE EXAMPLES")
    print("="*60)
    print("1. Single image inference")
    print("2. Batch inference") 
    print("3. Quick inference")
    print("4. Run all examples")
    print("="*60)
    
    try:
        choice = input("\nSelect an option (1-4): ").strip()
        print()
        
        if choice == "1":
            example_single_inference()
        elif choice == "2":
            example_batch_inference()
        elif choice == "3":
            example_quick_inference()
        elif choice == "4":
            print("🔄 Running all examples...\n")
            example_single_inference()
            example_batch_inference()
            example_quick_inference()
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check for available checkpoints
    possible_checkpoints = [
        "outputs/mask2former_iris/checkpoints/best.pt",
        "outputs/segformer_iris_a100/checkpoints/best.pt",
        "outputs/segformer_iris/checkpoints/best.pt",
    ]
    
    print("\n" + "="*60)
    print("🔍 CHECKING FOR TRAINED MODELS")
    print("="*60)
    
    found_checkpoints = []
    for checkpoint_path in possible_checkpoints:
        if os.path.exists(checkpoint_path):
            found_checkpoints.append(checkpoint_path)
            print(f"✅ Found: {checkpoint_path}")
        else:
            print(f"❌ Not found: {checkpoint_path}")
    
    print("="*60)
    
    if found_checkpoints:
        print(f"\n✅ Found {len(found_checkpoints)} trained model(s)")
        print(f"📋 Will use: {found_checkpoints[0]}")
        main()
    else:
        print("\n❌ No trained models found!")
        print("\nTo use this script:")
        print("1. Train a model using train_mask2former.py")
        print("2. Ensure the checkpoint file exists")
        print("3. Run this script again")
        print("\nExpected checkpoint locations:")
        for path in possible_checkpoints:
            print(f"  - {path}")
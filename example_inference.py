#!/usr/bin/env python3
"""
Example script showing how to use the iris segmentation inference module
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from inference import IrisSegmentationInference, quick_inference
import torch


def example_single_inference():
    """Example of running inference on a single image"""
    
    # Path to your trained model checkpoint
    checkpoint_path = "outputs/segformer_iris_a100/checkpoints/best.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        print("Please ensure you have trained the model and the checkpoint exists.")
        return
    
    # Find a sample image from the dataset
    sample_images = list(Path("dataset/images").glob("*.png"))[:3] if Path("dataset/images").exists() else []
    
    if not sample_images:
        print("❌ No sample images found in dataset/images/")
        print("Please ensure your dataset is available or provide an image path.")
        return
    
    print(f"🔍 Running inference on {len(sample_images)} sample images...")
    
    # Load the inference model
    model = IrisSegmentationInference(checkpoint_path)
    
    # Process each sample image
    for i, image_path in enumerate(sample_images):
        print(f"\n📸 Processing: {image_path.name}")
        
        try:
            # Run inference
            results = model.predict(image_path)
            
            # Print some statistics
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
                save_components=True
            )
            
        except Exception as e:
            print(f"   ❌ Error processing {image_path.name}: {e}")
    
    print(f"\n✅ Inference completed! Results saved to inference_results/")


def example_batch_inference():
    """Example of running batch inference"""
    
    checkpoint_path = "outputs/segformer_iris_a100/checkpoints/best.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return
    
    # Get multiple images
    sample_images = list(Path("dataset/images").glob("*.png"))[:5] if Path("dataset/images").exists() else []
    
    if len(sample_images) < 2:
        print("❌ Need at least 2 images for batch inference example")
        return
    
    print(f"🔍 Running batch inference on {len(sample_images)} images...")
    
    # Load model
    model = IrisSegmentationInference(checkpoint_path)
    
    # Run batch inference
    results_list = model.predict_batch(sample_images)
    
    # Process results
    for i, (image_path, results) in enumerate(zip(sample_images, results_list)):
        seg_mask = results['segmentation']['mask']
        iris_coverage = (seg_mask == 1).sum() / seg_mask.size * 100
        print(f"   📸 {image_path.name}: {iris_coverage:.1f}% iris coverage")
    
    print("✅ Batch inference completed!")


def example_quick_inference():
    """Example using the quick inference function"""
    
    checkpoint_path = "outputs/segformer_iris_a100/checkpoints/best.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return
    
    # Find one sample image
    sample_images = list(Path("dataset/images").glob("*.png"))[:1] if Path("dataset/images").exists() else []
    
    if not sample_images:
        print("❌ No sample images found")
        return
    
    image_path = sample_images[0]
    output_path = "quick_inference_result.png"
    
    print(f"🚀 Running quick inference on {image_path.name}...")
    
    try:
        # Run quick inference with visualization
        results = quick_inference(
            image_path=str(image_path),
            checkpoint_path=checkpoint_path,
            output_path="quick_inference_results",
            show_result=False  # Set to True to display results
        )
        
        seg_mask = results['segmentation']['mask']
        iris_coverage = (seg_mask == 1).sum() / seg_mask.size * 100
        print(f"✅ Quick inference completed! Iris coverage: {iris_coverage:.1f}%")
        
    except Exception as e:
        print(f"❌ Quick inference failed: {e}")


def main():
    """Main function with menu"""
    
    print("🔬 Iris Segmentation Inference Examples")
    print("=" * 50)
    print("1. Single image inference")
    print("2. Batch inference") 
    print("3. Quick inference")
    print("4. Run all examples")
    
    try:
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == "1":
            example_single_inference()
        elif choice == "2":
            example_batch_inference()
        elif choice == "3":
            example_quick_inference()
        elif choice == "4":
            print("\n🔄 Running all examples...")
            example_single_inference()
            print("\n" + "="*50)
            example_batch_inference()
            print("\n" + "="*50)
            example_quick_inference()
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Check if we have the required checkpoint
    checkpoint_path = "outputs/segformer_iris_a100/checkpoints/best.pt"
    
    print(f"🔍 Checking for trained model at: {checkpoint_path}")
    
    if os.path.exists(checkpoint_path):
        print("✅ Model checkpoint found!")
        main()
    else:
        print("❌ Model checkpoint not found!")
        print(f"Expected location: {checkpoint_path}")
        print("\nTo use this script:")
        print("1. Make sure you have trained the model")
        print("2. Verify the checkpoint path is correct")
        print("3. Update the checkpoint_path variable if needed")

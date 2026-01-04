#!/usr/bin/env python3
"""
Command line interface for iris segmentation inference
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# ✅ FIX: Import from src.inference and src.utils
from src.inference.inference import IrisSegmentationInference, quick_inference
from src.utils.visualization import visualize_prediction


def main():
    parser = argparse.ArgumentParser(
        description='Iris Segmentation Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer.py --image path/to/image.jpg --output results/
  python infer.py --image path/to/image.jpg --checkpoint custom_model.pt
  python infer.py --batch dataset/test_images/ --output batch_results/
  python infer.py --image path/to/image.jpg --show
        """
    )
    
    # Input options
    parser.add_argument('--image', type=str, help='Input image path')
    parser.add_argument('--batch', type=str, help='Input directory for batch processing')
    
    # Model options
    parser.add_argument('--checkpoint', type=str, 
                       default='outputs/segformer_iris_a100/checkpoints/best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    # Output options
    parser.add_argument('--output', type=str, default='inference_results',
                       help='Output directory or file path')
    parser.add_argument('--show', action='store_true', 
                       help='Display results using matplotlib')
    parser.add_argument('--save-components', action='store_true', default=True,
                       help='Save individual prediction components')
    parser.add_argument('--save-overlay', action='store_true', default=True,
                       help='Save overlay visualization on original image')
    parser.add_argument('--save-comparison', action='store_true', default=True,
                       help='Save side-by-side comparison visualization')
    
    # Overlay visualization options
    parser.add_argument('--iris-color', type=str, default='255,0,0',
                       help='RGB color for iris overlay (default: red)')
    parser.add_argument('--boundary-color', type=str, default='0,255,255',
                       help='RGB color for boundary overlay (default: cyan)')
    parser.add_argument('--iris-alpha', type=float, default=0.4,
                       help='Transparency for iris overlay (0.0-1.0)')
    parser.add_argument('--boundary-alpha', type=float, default=0.8,
                       help='Transparency for boundary overlay (0.0-1.0)')
    
    # Processing options
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                       help='Confidence threshold for boundary predictions')
    parser.add_argument('--no-boundary', action='store_true',
                       help='Skip boundary prediction processing')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.batch:
        print("❌ Error: Must specify either --image or --batch")
        parser.print_help()
        return
    
    if args.image and args.batch:
        print("❌ Error: Cannot specify both --image and --batch")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found at {args.checkpoint}")
        return
    
    try:
        if args.image:
            # Single image inference
            single_image_inference(args)
        else:
            # Batch inference
            batch_inference(args)
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()


def single_image_inference(args):
    """Process single image"""
    if not os.path.exists(args.image):
        print(f"❌ Error: Image not found at {args.image}")
        return
    
    print(f"🔍 Processing image: {args.image}")
    print(f"📋 Using checkpoint: {args.checkpoint}")
    
    # Load model
    print("⏳ Loading model...")
    model = IrisSegmentationInference(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Run inference
    print("🚀 Running inference...")
    results = model.predict(
        args.image,
        return_boundary=not args.no_boundary,
        confidence_threshold=args.confidence_threshold
    )
    
    # Print statistics
    seg_mask = results['segmentation']['mask']
    iris_coverage = (seg_mask == 1).sum() / seg_mask.size * 100
    avg_confidence = results['segmentation']['confidence'].mean()
    
    print(f"📊 Results:")
    print(f"   • Image size: {seg_mask.shape}")
    print(f"   • Iris coverage: {iris_coverage:.1f}%")
    print(f"   • Average confidence: {avg_confidence:.3f}")
    
    if 'boundary' in results:
        boundary_density = results['boundary']['boundary_mask'].sum() / results['boundary']['boundary_mask'].size * 100
        print(f"   • Boundary density: {boundary_density:.1f}%")
    
    # Parse overlay colors
    iris_color = tuple(map(int, args.iris_color.split(',')))
    boundary_color = tuple(map(int, args.boundary_color.split(',')))
    
    overlay_kwargs = {
        'iris_color': iris_color,
        'boundary_color': boundary_color,
        'iris_alpha': args.iris_alpha,
        'boundary_alpha': args.boundary_alpha,
        'show_boundary': not args.no_boundary
    }
    
    # Save results
    print(f"💾 Saving results to: {args.output}")
    model.save_prediction(
        results, 
        args.output,
        original_image=args.image,
        save_overlay=args.save_overlay,
        save_comparison=args.save_comparison,
        save_components=args.save_components,
        overlay_kwargs=overlay_kwargs
    )
    
    # Show results if requested
    if args.show:
        try:
            import matplotlib.pyplot as plt
            visualize_prediction(args.image, results)
            plt.show()
        except ImportError:
            print("⚠️ Cannot display results: matplotlib not available")
    
    print("✅ Single image inference completed!")


def batch_inference(args):
    """Process batch of images"""
    input_dir = Path(args.batch)
    
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"❌ Error: Directory not found at {args.batch}")
        return
    
    # Find images
    image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
        image_files.extend(input_dir.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"❌ Error: No images found in {args.batch}")
        return
    
    print(f"🔍 Processing {len(image_files)} images from: {args.batch}")
    print(f"📋 Using checkpoint: {args.checkpoint}")
    
    # Load model
    print("⏳ Loading model...")
    model = IrisSegmentationInference(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse overlay colors
    iris_color = tuple(map(int, args.iris_color.split(',')))
    boundary_color = tuple(map(int, args.boundary_color.split(',')))
    
    overlay_kwargs = {
        'iris_color': iris_color,
        'boundary_color': boundary_color,
        'iris_alpha': args.iris_alpha,
        'boundary_alpha': args.boundary_alpha,
        'show_boundary': not args.no_boundary
    }
    
    # Process each image
    print("🚀 Running batch inference...")
    results_summary = []
    
    for i, image_path in enumerate(image_files, 1):
        try:
            print(f"   📸 [{i}/{len(image_files)}] Processing: {image_path.name}")
            
            # Run inference
            results = model.predict(
                image_path,
                return_boundary=not args.no_boundary,
                confidence_threshold=args.confidence_threshold
            )
            
            # Calculate statistics
            seg_mask = results['segmentation']['mask']
            iris_coverage = (seg_mask == 1).sum() / seg_mask.size * 100
            avg_confidence = results['segmentation']['confidence'].mean()
            
            # Save results
            result_name = f"result_{i:03d}_{image_path.stem}"
            model.save_prediction(
                results,
                output_dir / result_name,
                original_image=image_path,
                save_overlay=args.save_overlay,
                save_comparison=args.save_comparison,
                save_components=args.save_components,
                overlay_kwargs=overlay_kwargs
            )
            
            # Store summary
            summary = {
                'image': image_path.name,
                'iris_coverage': iris_coverage,
                'confidence': avg_confidence
            }
            
            if 'boundary' in results:
                boundary_density = results['boundary']['boundary_mask'].sum() / results['boundary']['boundary_mask'].size * 100
                summary['boundary_density'] = boundary_density
            
            results_summary.append(summary)
            
            print(f"      ✓ Iris: {iris_coverage:.1f}%, Confidence: {avg_confidence:.3f}")
            
        except Exception as e:
            print(f"      ❌ Error processing {image_path.name}: {e}")
    
    # Save summary report
    summary_path = output_dir / 'batch_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Batch Inference Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total images processed: {len(results_summary)}\n")
        f.write(f"Model checkpoint: {args.checkpoint}\n")
        f.write(f"Output directory: {output_dir}\n\n")
        
        f.write("Individual Results:\n")
        f.write("-" * 50 + "\n")
        for result in results_summary:
            f.write(f"Image: {result['image']}\n")
            f.write(f"  Iris Coverage: {result['iris_coverage']:.1f}%\n")
            f.write(f"  Confidence: {result['confidence']:.3f}\n")
            if 'boundary_density' in result:
                f.write(f"  Boundary Density: {result['boundary_density']:.1f}%\n")
            f.write("\n")
        
        # Statistics
        if results_summary:
            avg_coverage = sum(r['iris_coverage'] for r in results_summary) / len(results_summary)
            avg_conf = sum(r['confidence'] for r in results_summary) / len(results_summary)
            
            f.write("Overall Statistics:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Average iris coverage: {avg_coverage:.1f}%\n")
            f.write(f"Average confidence: {avg_conf:.3f}\n")
    
    print(f"\n✅ Batch inference completed!")
    print(f"📊 Processed {len(results_summary)} images")
    print(f"📝 Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
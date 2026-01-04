#!/usr/bin/env python3
"""
FIXED Command line interface for Mask2Former iris segmentation inference
Key fixes:
1. Import Mask2FormerInference (correct class name)
2. Updated all function calls
3. Better error handling
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.inference import Mask2FormerInference  # FIXED: Correct class name
from src.utils.visualization import visualize_prediction


def main():
    parser = argparse.ArgumentParser(
        description='Mask2Former Iris Segmentation Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image inference
  python infer_mask2former.py --image path/to/image.jpg --output results/
  
  # Batch inference
  python infer_mask2former.py --batch dataset/test_images/ --output batch_results/
  
  # Custom visualization
  python infer_mask2former.py --image path/to/image.jpg --iris-color "0,255,0" --show
  
  # Quick inference with default settings
  python infer_mask2former.py --image path/to/image.jpg
        """
    )
    
    # Input options
    parser.add_argument('--image', type=str, help='Input image path')
    parser.add_argument('--batch', type=str, help='Input directory for batch processing')
    
    # Model options
    parser.add_argument('--checkpoint', type=str, 
                       default='outputs/mask2former_iris/checkpoints/best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    # Output options
    parser.add_argument('--output', type=str, default='mask2former_inference_results',
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
        print(f"\nPlease ensure:")
        print(f"1. You have trained the model")
        print(f"2. The checkpoint path is correct")
        print(f"3. The checkpoint file exists")
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
    """FIXED: Process single image"""
    if not os.path.exists(args.image):
        print(f"❌ Error: Image not found at {args.image}")
        return
    
    print(f"{'='*60}")
    print(f"MASK2FORMER SINGLE IMAGE INFERENCE")
    print(f"{'='*60}")
    print(f"📸 Image: {args.image}")
    print(f"📋 Checkpoint: {args.checkpoint}")
    print(f"💾 Output: {args.output}")
    print(f"{'='*60}\n")
    
    # Load model
    print("⏳ Loading Mask2Former model...")
    try:
        model = Mask2FormerInference(  # FIXED: Correct class name
            checkpoint_path=args.checkpoint,
            device=args.device
        )
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Run inference
    print("🚀 Running inference...")
    try:
        results = model.predict(
            args.image,
            return_boundary=not args.no_boundary,
            confidence_threshold=args.confidence_threshold
        )
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return
    
    # Print statistics
    seg_mask = results['segmentation']['mask']
    iris_coverage = (seg_mask == 1).sum() / seg_mask.size * 100
    avg_confidence = results['segmentation']['confidence'].mean()
    
    print(f"\n📊 Results:")
    print(f"   • Image size: {seg_mask.shape}")
    print(f"   • Iris coverage: {iris_coverage:.1f}%")
    print(f"   • Average confidence: {avg_confidence:.3f}")
    
    if 'boundary' in results:
        boundary_density = results['boundary']['boundary_mask'].sum() / results['boundary']['boundary_mask'].size * 100
        print(f"   • Boundary density: {boundary_density:.1f}%")
    
    # Parse overlay colors
    try:
        iris_color = tuple(map(int, args.iris_color.split(',')))
        boundary_color = tuple(map(int, args.boundary_color.split(',')))
    except:
        print("⚠️  Invalid color format, using defaults")
        iris_color = (255, 0, 0)
        boundary_color = (0, 255, 255)
    
    overlay_kwargs = {
        'iris_color': iris_color,
        'boundary_color': boundary_color,
        'iris_alpha': args.iris_alpha,
        'boundary_alpha': args.boundary_alpha,
        'show_boundary': not args.no_boundary
    }
    
    # Save results
    print(f"\n💾 Saving results...")
    try:
        model.save_prediction(
            results, 
            args.output,
            original_image=args.image,
            save_overlay=args.save_overlay,
            save_comparison=args.save_comparison,
            save_components=args.save_components,
            overlay_kwargs=overlay_kwargs
        )
    except Exception as e:
        print(f"⚠️  Error saving results: {e}")
    
    # Show results if requested
    if args.show:
        try:
            import matplotlib.pyplot as plt
            visualize_prediction(args.image, results)
            plt.show()
        except ImportError:
            print("⚠️ Cannot display results: matplotlib not available")
        except Exception as e:
            print(f"⚠️  Error displaying results: {e}")
    
    print(f"\n✅ Single image inference completed!")
    print(f"{'='*60}\n")


def batch_inference(args):
    """FIXED: Process batch of images"""
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
    
    print(f"{'='*60}")
    print(f"MASK2FORMER BATCH INFERENCE")
    print(f"{'='*60}")
    print(f"📁 Input: {args.batch}")
    print(f"📸 Images: {len(image_files)}")
    print(f"📋 Checkpoint: {args.checkpoint}")
    print(f"💾 Output: {args.output}")
    print(f"{'='*60}\n")
    
    # Load model
    print("⏳ Loading Mask2Former model...")
    try:
        model = Mask2FormerInference(  # FIXED: Correct class name
            checkpoint_path=args.checkpoint,
            device=args.device
        )
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse overlay colors
    try:
        iris_color = tuple(map(int, args.iris_color.split(',')))
        boundary_color = tuple(map(int, args.boundary_color.split(',')))
    except:
        print("⚠️  Invalid color format, using defaults")
        iris_color = (255, 0, 0)
        boundary_color = (0, 255, 255)
    
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
    failed_images = []
    
    for i, image_path in enumerate(image_files, 1):
        try:
            print(f"   📸 [{i}/{len(image_files)}] Processing: {image_path.name}...", end=" ")
            
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
            
            print(f"✓ Iris: {iris_coverage:.1f}%, Conf: {avg_confidence:.3f}")
            
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
            failed_images.append({
                'image': image_path.name,
                'error': str(e)
            })
    
    # Save summary report
    print(f"\n💾 Saving summary report...")
    summary_path = output_dir / 'batch_summary.txt'
    
    try:
        with open(summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("MASK2FORMER BATCH INFERENCE SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total images: {len(image_files)}\n")
            f.write(f"Successfully processed: {len(results_summary)}\n")
            f.write(f"Failed: {len(failed_images)}\n")
            f.write(f"Model checkpoint: {args.checkpoint}\n")
            f.write(f"Output directory: {output_dir}\n\n")
            
            if results_summary:
                f.write("="*60 + "\n")
                f.write("SUCCESSFUL RESULTS\n")
                f.write("="*60 + "\n\n")
                for result in results_summary:
                    f.write(f"Image: {result['image']}\n")
                    f.write(f"  Iris Coverage: {result['iris_coverage']:.1f}%\n")
                    f.write(f"  Confidence: {result['confidence']:.3f}\n")
                    if 'boundary_density' in result:
                        f.write(f"  Boundary Density: {result['boundary_density']:.1f}%\n")
                    f.write("\n")
            
            if failed_images:
                f.write("="*60 + "\n")
                f.write("FAILED IMAGES\n")
                f.write("="*60 + "\n\n")
                for fail in failed_images:
                    f.write(f"Image: {fail['image']}\n")
                    f.write(f"  Error: {fail['error']}\n\n")
            
            # Statistics
            if results_summary:
                avg_coverage = sum(r['iris_coverage'] for r in results_summary) / len(results_summary)
                avg_conf = sum(r['confidence'] for r in results_summary) / len(results_summary)
                
                f.write("="*60 + "\n")
                f.write("OVERALL STATISTICS\n")
                f.write("="*60 + "\n\n")
                f.write(f"Average iris coverage: {avg_coverage:.1f}%\n")
                f.write(f"Average confidence: {avg_conf:.3f}\n")
                
                if any('boundary_density' in r for r in results_summary):
                    boundary_densities = [r['boundary_density'] for r in results_summary if 'boundary_density' in r]
                    avg_boundary = sum(boundary_densities) / len(boundary_densities)
                    f.write(f"Average boundary density: {avg_boundary:.1f}%\n")
        
        print(f"✅ Summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"⚠️  Error saving summary: {e}")
    
    # Print final statistics
    print(f"\n{'='*60}")
    print(f"BATCH INFERENCE COMPLETED")
    print(f"{'='*60}")
    print(f"✅ Processed: {len(results_summary)}/{len(image_files)}")
    if failed_images:
        print(f"❌ Failed: {len(failed_images)}/{len(image_files)}")
    print(f"📁 Results: {output_dir}")
    print(f"📝 Summary: {summary_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
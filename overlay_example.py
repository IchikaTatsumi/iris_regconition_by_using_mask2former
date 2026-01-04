#!/usr/bin/env python3
"""
Comprehensive example of overlay visualization for iris segmentation
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from inference import IrisSegmentationInference
from utils.visualization import create_overlay_visualization, create_comparison_visualization
from PIL import Image
import numpy as np


def create_overlay_examples():
    """Create various overlay examples with different settings"""
    
    checkpoint_path = "outputs/segformer_iris_a100/checkpoints/best.pt"
    sample_image = "dataset/images/C100_S1_I10.png"
    
    if not os.path.exists(checkpoint_path) or not os.path.exists(sample_image):
        print("❌ Required files not found")
        return
    
    print("🎨 Creating Overlay Visualization Examples")
    print("=" * 50)
    
    # Load model and run inference
    model = IrisSegmentationInference(checkpoint_path)
    results = model.predict(sample_image)
    
    # Create output directory
    output_dir = Path("overlay_examples")
    output_dir.mkdir(exist_ok=True)
    
    # Example configurations
    overlay_configs = [
        {
            "name": "clinical_red",
            "description": "Clinical style with red iris",
            "config": {
                "iris_color": (255, 0, 0),
                "boundary_color": (255, 255, 255),
                "iris_alpha": 0.4,
                "boundary_alpha": 0.9,
                "boundary_thickness": 1
            }
        },
        {
            "name": "subtle_blue",
            "description": "Subtle blue overlay",
            "config": {
                "iris_color": (0, 100, 255),
                "boundary_color": (255, 255, 0),
                "iris_alpha": 0.3,
                "boundary_alpha": 0.7,
                "boundary_thickness": 2
            }
        },
        {
            "name": "high_contrast",
            "description": "High contrast visualization",
            "config": {
                "iris_color": (255, 0, 255),
                "boundary_color": (0, 255, 0),
                "iris_alpha": 0.6,
                "boundary_alpha": 0.95,
                "boundary_thickness": 3
            }
        },
        {
            "name": "scientific",
            "description": "Scientific publication style",
            "config": {
                "iris_color": (0, 255, 0),
                "boundary_color": (255, 100, 0),
                "iris_alpha": 0.35,
                "boundary_alpha": 0.8,
                "boundary_thickness": 2
            }
        },
        {
            "name": "iris_only",
            "description": "Iris segmentation only",
            "config": {
                "iris_color": (128, 0, 128),
                "iris_alpha": 0.5,
                "show_boundary": False
            }
        }
    ]
    
    print(f"🖼️ Creating {len(overlay_configs)} overlay examples...")
    
    for i, config in enumerate(overlay_configs, 1):
        print(f"   {i}️⃣ {config['description']}")
        
        # Create overlay
        overlay = create_overlay_visualization(sample_image, results, **config['config'])
        
        # Save overlay
        output_path = output_dir / f"{config['name']}_overlay.png"
        Image.fromarray(overlay).save(output_path)
        print(f"      💾 Saved: {output_path}")
    
    # Create comprehensive comparison
    print("📊 Creating comprehensive comparison...")
    create_comparison_visualization(
        sample_image, 
        results, 
        output_dir / "comprehensive_comparison.png",
        figsize=(18, 6)
    )
    
    # Create a grid of all overlays for comparison
    print("🗂️ Creating overlay grid comparison...")
    create_overlay_grid(sample_image, results, overlay_configs, output_dir)
    
    print("\n✅ All overlay examples created!")
    print(f"📁 Check the results in: {output_dir}/")
    
    # Print statistics
    seg_mask = results['segmentation']['mask']
    iris_coverage = (seg_mask == 1).sum() / seg_mask.size * 100
    avg_confidence = results['segmentation']['confidence'].mean()
    
    print(f"\n📊 Segmentation Statistics:")
    print(f"   • Iris coverage: {iris_coverage:.1f}%")
    print(f"   • Average confidence: {avg_confidence:.3f}")
    
    if 'boundary' in results:
        boundary_density = results['boundary']['boundary_mask'].sum() / results['boundary']['boundary_mask'].size * 100
        print(f"   • Boundary density: {boundary_density:.1f}%")


def create_overlay_grid(image_path, results, configs, output_dir):
    """Create a grid showing all overlay styles"""
    import matplotlib.pyplot as plt
    
    # Load original image
    orig_image = np.array(Image.open(image_path))
    
    # Create figure with grid
    n_configs = len(configs)
    n_cols = 3
    n_rows = (n_configs + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    
    # Handle case where we have only one row
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, config in enumerate(configs):
        row = i // n_cols
        col = i % n_cols
        
        # Create overlay
        overlay = create_overlay_visualization(image_path, results, **config['config'])
        
        # Plot
        axes[row, col].imshow(overlay)
        axes[row, col].set_title(f"{config['name'].replace('_', ' ').title()}\n{config['description']}", 
                                fontsize=10, fontweight='bold')
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(n_configs, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('Iris Segmentation Overlay Styles Comparison', fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    # Save grid
    grid_path = output_dir / "overlay_styles_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"🗂️ Overlay grid saved: {grid_path}")


def demonstrate_cli_usage():
    """Show CLI usage examples for overlay functionality"""
    
    print("\n🖥️ Command Line Usage Examples:")
    print("=" * 50)
    
    examples = [
        {
            "description": "Basic overlay with default red iris",
            "command": "python infer.py --image path/to/image.jpg --output results/"
        },
        {
            "description": "Green iris with yellow boundary",
            "command": "python infer.py --image path/to/image.jpg --iris-color \"0,255,0\" --boundary-color \"255,255,0\""
        },
        {
            "description": "High transparency iris (subtle)",
            "command": "python infer.py --image path/to/image.jpg --iris-alpha 0.2 --boundary-alpha 0.6"
        },
        {
            "description": "Clinical style visualization",
            "command": "python infer.py --image path/to/image.jpg --iris-color \"255,0,0\" --boundary-color \"255,255,255\" --iris-alpha 0.4"
        },
        {
            "description": "Skip boundary visualization",
            "command": "python infer.py --image path/to/image.jpg --no-boundary"
        },
        {
            "description": "Only save overlay (no components)",
            "command": "python infer.py --image path/to/image.jpg --save-components False --save-overlay --save-comparison"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['description']}:")
        print(f"   {example['command']}")
        print()


if __name__ == "__main__":
    create_overlay_examples()
    demonstrate_cli_usage()

"""
FIXED Inference module for Mask2Former iris segmentation
Key fixes:
1. Correct class name (Mask2FormerInference, not IrisSegmentationInference)
2. Proper model loading
3. Simplified post-processing for semantic segmentation
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple
import cv2
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mask2former import create_model
from data.transforms import get_inference_transform


class Mask2FormerInference:
    """
    FIXED: Inference class for Mask2Former iris segmentation
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize inference model
        
        Args:
            checkpoint_path: Path to trained checkpoint
            device: Device ('cuda' or 'cpu')
            model_config: Model configuration
        """
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Default model configuration
        if model_config is None:
            model_config = {
                'architecture': 'mask2former',
                'model_name': 'facebook/mask2former-swin-small-coco-panoptic',
                'model_type': 'enhanced',
                'num_labels': 2,
                'add_boundary_head': True,
                'num_queries': 50
            }
        self.model_config = model_config
        
        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Get transforms
        self.transform = get_inference_transform()
        
        print(f"✅ Mask2Former loaded from {checkpoint_path}")
        print(f"📱 Device: {self.device}")
    
    def _load_model(self) -> torch.nn.Module:
        """FIXED: Load trained model from checkpoint"""
        try:
            # Load checkpoint
            checkpoint = torch.load(
                self.checkpoint_path, 
                map_location='cpu',
                weights_only=False
            )
            
            # Create model
            model = create_model(**self.model_config)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"📊 Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
                if 'best_metric' in checkpoint:
                    print(f"🏆 Best metric: {checkpoint['best_metric']:.4f}")
            else:
                model.load_state_dict(checkpoint)
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def preprocess_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        target_size: int = 512
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for inference
        
        Args:
            image: Input image
            target_size: Target size
        
        Returns:
            (preprocessed_tensor, original_size)
        """
        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Store original size
        original_size = image.size  # (width, height)
        
        # Apply transforms
        pixel_values = self.transform(image)
        pixel_values = pixel_values.unsqueeze(0)  # Add batch dimension
        
        return pixel_values, original_size
    
    def postprocess_prediction(
        self,
        logits: torch.Tensor,
        original_size: Tuple[int, int],
        apply_softmax: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        FIXED: Postprocess model predictions
        
        Args:
            logits: [1, num_classes, H, W]
            original_size: (width, height)
            apply_softmax: Whether to apply softmax
        
        Returns:
            Dictionary with prediction results
        """
        logits = logits.cpu()
        
        # Resize to original size
        logits_resized = F.interpolate(
            logits,
            size=(original_size[1], original_size[0]),  # (H, W)
            mode='bilinear',
            align_corners=False
        )
        
        # Get probabilities
        if apply_softmax:
            probs = F.softmax(logits_resized, dim=1)
        else:
            probs = logits_resized
        
        # Get prediction mask
        pred_mask = torch.argmax(probs, dim=1)
        
        # Convert to numpy
        pred_mask = pred_mask.squeeze(0).numpy().astype(np.uint8)
        probs = probs.squeeze(0).numpy()
        
        # Get iris probability
        iris_prob = probs[1]
        
        return {
            'mask': pred_mask,
            'probabilities': probs,
            'iris_probability': iris_prob,
            'confidence': np.max(probs, axis=0)
        }
    
    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        return_boundary: bool = True,
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        FIXED: Perform inference on single image
        
        Args:
            image: Input image
            return_boundary: Whether to return boundary
            confidence_threshold: Threshold for boundary
        
        Returns:
            Prediction results dictionary
        """
        # Preprocess
        pixel_values, original_size = self.preprocess_image(image)
        pixel_values = pixel_values.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(
                pixel_values, 
                return_boundary=return_boundary
            )
        
        # Postprocess segmentation
        seg_results = self.postprocess_prediction(
            outputs['logits'],
            original_size
        )
        
        # Process boundary if available
        boundary_results = None
        if 'boundary_logits' in outputs:
            boundary_logits = outputs['boundary_logits'].cpu()
            boundary_resized = F.interpolate(
                boundary_logits,
                size=(original_size[1], original_size[0]),
                mode='bilinear',
                align_corners=False
            )
            boundary_prob = torch.sigmoid(boundary_resized).squeeze().numpy()
            boundary_mask = (boundary_prob > confidence_threshold).astype(np.uint8)
            
            boundary_results = {
                'boundary_probability': boundary_prob,
                'boundary_mask': boundary_mask
            }
        
        # Combine results
        results = {
            'segmentation': seg_results,
            'original_size': original_size,
            'input_size': (pixel_values.shape[-1], pixel_values.shape[-2])
        }
        
        if boundary_results:
            results['boundary'] = boundary_results
        
        return results
    
    def predict_batch(
        self,
        images: list,
        return_boundary: bool = True,
        confidence_threshold: float = 0.5
    ) -> list:
        """
        Batch inference
        
        Args:
            images: List of images
            return_boundary: Whether to return boundary
            confidence_threshold: Threshold
        
        Returns:
            List of prediction results
        """
        results = []
        for image in images:
            result = self.predict(
                image,
                return_boundary=return_boundary,
                confidence_threshold=confidence_threshold
            )
            results.append(result)
        
        return results
    
    def save_prediction(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
        original_image: Optional[Union[str, Path, Image.Image, np.ndarray]] = None,
        save_overlay: bool = True,
        save_comparison: bool = True,
        save_components: bool = True,
        overlay_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Save prediction results
        
        Args:
            results: Prediction results
            output_path: Output path
            original_image: Original image for overlay
            save_overlay: Save overlay
            save_comparison: Save comparison
            save_components: Save components
            overlay_kwargs: Overlay arguments
        """
        output_path = Path(output_path)
        
        if output_path.is_dir():
            output_dir = output_path
            base_name = "prediction"
        else:
            output_dir = output_path.parent
            base_name = output_path.stem
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if save_components:
            # Save mask
            seg_mask = results['segmentation']['mask']
            seg_mask_img = Image.fromarray((seg_mask * 255).astype(np.uint8))
            seg_mask_img.save(output_dir / f"{base_name}_mask.png")
            
            # Save probability
            iris_prob = results['segmentation']['iris_probability']
            iris_prob_img = Image.fromarray((iris_prob * 255).astype(np.uint8))
            iris_prob_img.save(output_dir / f"{base_name}_iris_prob.png")
            
            # Save boundary
            if 'boundary' in results:
                boundary_prob = results['boundary']['boundary_probability']
                boundary_img = Image.fromarray((boundary_prob * 255).astype(np.uint8))
                boundary_img.save(output_dir / f"{base_name}_boundary.png")
        
        # Save overlays if original image provided
        if original_image is not None and (save_overlay or save_comparison):
            try:
                # Import visualization utilities
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from utils.visualization import (
                    create_overlay_visualization,
                    create_comparison_visualization
                )
                
                overlay_kwargs = overlay_kwargs or {}
                
                if save_overlay:
                    overlay_path = output_dir / f"{base_name}_overlay.png"
                    overlay = create_overlay_visualization(
                        original_image, results, **overlay_kwargs
                    )
                    Image.fromarray(overlay).save(overlay_path)
                
                if save_comparison:
                    comparison_path = output_dir / f"{base_name}_comparison.png"
                    create_comparison_visualization(
                        original_image, results, comparison_path
                    )
            except ImportError as e:
                print(f"⚠️  Could not create visualizations: {e}")
        
        print(f"💾 Predictions saved to {output_dir}")
    
    def create_overlay(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        results: Dict[str, Any],
        **overlay_kwargs
    ) -> np.ndarray:
        """Create overlay visualization"""
        try:
            from utils.visualization import create_overlay_visualization
            return create_overlay_visualization(image, results, **overlay_kwargs)
        except ImportError:
            print("⚠️  Visualization utilities not available")
            return None


def load_inference_model(checkpoint_path: str, **kwargs) -> Mask2FormerInference:
    """
    FIXED: Convenience function to load model
    
    Args:
        checkpoint_path: Path to checkpoint
        **kwargs: Additional arguments
    
    Returns:
        Loaded inference model
    """
    return Mask2FormerInference(checkpoint_path, **kwargs)


def quick_inference(
    image_path: str,
    checkpoint_path: str,
    output_path: Optional[str] = None,
    show_result: bool = False
) -> Dict[str, Any]:
    """
    FIXED: Quick inference function
    
    Args:
        image_path: Input image path
        checkpoint_path: Model checkpoint path
        output_path: Output path
        show_result: Whether to show result
    
    Returns:
        Prediction results
    """
    # Load model
    model = load_inference_model(checkpoint_path)
    
    # Predict
    results = model.predict(image_path)
    
    # Save results
    if output_path:
        model.save_prediction(
            results,
            output_path,
            original_image=image_path,
            save_overlay=True,
            save_comparison=True,
            save_components=True
        )
    
    # Show result
    if show_result:
        try:
            import matplotlib.pyplot as plt
            from utils.visualization import visualize_prediction
            visualize_prediction(image_path, results)
            plt.show()
        except ImportError:
            print("⚠️  Cannot display: matplotlib not available")
    
    return results


if __name__ == "__main__":
    print("Testing FIXED Mask2Former inference...")
    
    # Test with dummy checkpoint path
    checkpoint_path = "outputs/mask2former_iris/checkpoints/best.pt"
    
    if Path(checkpoint_path).exists():
        try:
            model = Mask2FormerInference(checkpoint_path)
            print("✅ Model loaded successfully")
            
            # Test with sample image
            sample_image = "dataset/images/C100_S1_I10.png"
            if Path(sample_image).exists():
                results = model.predict(sample_image)
                print("✅ Inference test passed")
                print(f"   Mask shape: {results['segmentation']['mask'].shape}")
            else:
                print("⚠️  Sample image not found for testing")
                
        except Exception as e:
            print(f"⚠️  Test failed: {e}")
    else:
        print("⚠️  Checkpoint not found for testing")
    
    print("\n✅ Inference module test completed")
"""
Inference module for trained SegFormer iris segmentation model
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple
import cv2

from model.models import load_pretrained_iris_model, create_model
from data.transforms import get_inference_transform
from utils.visualization import (
    visualize_prediction, 
    create_overlay_visualization, 
    save_overlay_visualization,
    create_comparison_visualization
)


class IrisSegmentationInference:
    """
    Inference class for iris segmentation using trained SegFormer model
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
            checkpoint_path: Path to trained model checkpoint
            device: Device to use for inference ('cuda' or 'cpu')
            model_config: Model configuration dictionary
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
                'model_name': 'nvidia/segformer-b1-finetuned-ade-512-512',
                'model_type': 'enhanced',
                'num_labels': 2,
                'add_boundary_head': True
            }
        self.model_config = model_config
        
        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Get transforms
        self.transform = get_inference_transform()
        
        print(f"✅ Inference model loaded from {checkpoint_path}")
        print(f"📱 Device: {self.device}")
    
    def _load_model(self) -> torch.nn.Module:
        """Load the trained model from checkpoint"""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
            
            # Create model
            model = create_model(**self.model_config)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"📊 Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
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
            image: Input image (path, PIL Image, or numpy array)
            target_size: Target size for model input
        
        Returns:
            Tuple of (preprocessed_tensor, original_size)
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert BGR to RGB if OpenCV image
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
        Postprocess model predictions
        
        Args:
            logits: Model output logits [1, num_classes, H, W]
            original_size: Original image size (width, height)
            apply_softmax: Whether to apply softmax to get probabilities
        
        Returns:
            Dictionary containing prediction results
        """
        # Move to CPU
        logits = logits.cpu()
        
        # Resize to original image size
        logits_resized = F.interpolate(
            logits,
            size=(original_size[1], original_size[0]),  # PIL size is (W, H), tensor is (H, W)
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
        
        # Get iris probability (class 1)
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
        Perform inference on a single image
        
        Args:
            image: Input image
            return_boundary: Whether to return boundary predictions
            confidence_threshold: Confidence threshold for predictions
        
        Returns:
            Dictionary containing all prediction results
        """
        # Preprocess
        pixel_values, original_size = self.preprocess_image(image)
        pixel_values = pixel_values.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(pixel_values, return_boundary=return_boundary)
        
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
        Perform inference on a batch of images
        
        Args:
            images: List of input images
            return_boundary: Whether to return boundary predictions
            confidence_threshold: Confidence threshold for predictions
        
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
        Save prediction results to files
        
        Args:
            results: Prediction results from predict()
            output_path: Output directory or file path
            original_image: Original image for overlay (if not provided, overlay won't be saved)
            save_overlay: Whether to save overlay visualization
            save_comparison: Whether to save comparison visualization
            save_components: Whether to save individual components
            overlay_kwargs: Additional arguments for overlay visualization
        """
        output_path = Path(output_path)
        
        if output_path.is_dir():
            # Save to directory with default names
            output_dir = output_path
            base_name = "prediction"
        else:
            # Use provided filename
            output_dir = output_path.parent
            base_name = output_path.stem
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if save_components:
            # Save segmentation mask
            seg_mask = results['segmentation']['mask']
            seg_mask_img = Image.fromarray((seg_mask * 255).astype(np.uint8))
            seg_mask_img.save(output_dir / f"{base_name}_mask.png")
            
            # Save iris probability map
            iris_prob = results['segmentation']['iris_probability']
            iris_prob_img = Image.fromarray((iris_prob * 255).astype(np.uint8))
            iris_prob_img.save(output_dir / f"{base_name}_iris_prob.png")
            
            # Save boundary if available
            if 'boundary' in results:
                boundary_prob = results['boundary']['boundary_probability']
                boundary_img = Image.fromarray((boundary_prob * 255).astype(np.uint8))
                boundary_img.save(output_dir / f"{base_name}_boundary.png")
        
        # Save overlay visualizations if original image is provided
        if original_image is not None:
            overlay_kwargs = overlay_kwargs or {}
            
            if save_overlay:
                overlay_path = output_dir / f"{base_name}_overlay.png"
                save_overlay_visualization(original_image, results, overlay_path, **overlay_kwargs)
            
            if save_comparison:
                comparison_path = output_dir / f"{base_name}_comparison.png"
                create_comparison_visualization(original_image, results, comparison_path)
        
        print(f"💾 Predictions saved to {output_dir}")
    
    def create_overlay(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        results: Dict[str, Any],
        **overlay_kwargs
    ) -> np.ndarray:
        """
        Create overlay visualization of results on original image
        
        Args:
            image: Original image
            results: Prediction results from predict()
            **overlay_kwargs: Additional arguments for overlay visualization
        
        Returns:
            Overlay image as numpy array
        """
        return create_overlay_visualization(image, results, **overlay_kwargs)


def load_inference_model(checkpoint_path: str, **kwargs) -> IrisSegmentationInference:
    """
    Convenience function to load inference model
    
    Args:
        checkpoint_path: Path to model checkpoint
        **kwargs: Additional arguments for IrisSegmentationInference
    
    Returns:
        Loaded inference model
    """
    return IrisSegmentationInference(checkpoint_path, **kwargs)


def quick_inference(
    image_path: str,
    checkpoint_path: str,
    output_path: Optional[str] = None,
    show_result: bool = False
) -> Dict[str, Any]:
    """
    Quick inference function for single image
    
    Args:
        image_path: Path to input image
        checkpoint_path: Path to model checkpoint
        output_path: Output path for results (optional)
        show_result: Whether to display results
    
    Returns:
        Prediction results
    """
    # Load model
    model = load_inference_model(checkpoint_path)
    
    # Predict
    results = model.predict(image_path)
    
    # Save results if output path provided
    if output_path:
        model.save_prediction(results, output_path)
    
    # Show results if requested
    if show_result:
        try:
            import matplotlib.pyplot as plt
            visualize_prediction(image_path, results)
            plt.show()
        except ImportError:
            print("⚠️  Matplotlib not available for visualization")
    
    return results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Iris segmentation inference')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--show', action='store_true', help='Show results')
    parser.add_argument('--device', type=str, default=None, help='Device to use')
    
    args = parser.parse_args()
    
    # Run inference
    results = quick_inference(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        show_result=args.show
    )
    
    print("🎯 Inference completed!")
    print(f"📊 Iris coverage: {np.mean(results['segmentation']['mask']) * 100:.1f}%")
    if 'boundary' in results:
        boundary_density = np.mean(results['boundary']['boundary_mask']) * 100
        print(f"🔲 Boundary density: {boundary_density:.1f}%")

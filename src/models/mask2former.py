"""
Mask2Former model with boundary refinement for iris segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig
from typing import Optional, Dict, Any

from .heads import BoundaryRefinementHead


class EnhancedMask2Former(nn.Module):
    """
    Mask2Former with boundary refinement head for iris segmentation
    """
    
    def __init__(
        self,
        model_name: str = "facebook/mask2former-swin-small-coco-panoptic",
        num_labels: int = 2,
        add_boundary_head: bool = True,
        freeze_backbone: bool = False,
        freeze_epochs: int = 0,
        use_auxiliary_loss: bool = True
    ):
        super().__init__()
        
        # Load pretrained Mask2Former
        self.mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        
        # Add boundary refinement head
        self.add_boundary_head = add_boundary_head
        if add_boundary_head:
            self.boundary_head = BoundaryRefinementHead(
                in_channels=num_labels,
                hidden_channels=64
            )
        
        # Freezing configuration
        self.freeze_backbone = freeze_backbone
        self.freeze_epochs = freeze_epochs
        self.current_epoch = 0
        self.use_auxiliary_loss = use_auxiliary_loss
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze backbone parameters"""
        # Freeze pixel decoder and transformer decoder
        for param in self.mask2former.model.pixel_level_module.parameters():
            param.requires_grad = False
        print("Mask2Former backbone frozen")
    
    def _unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.mask2former.model.pixel_level_module.parameters():
            param.requires_grad = True
        print("Mask2Former backbone unfrozen")
    
    def set_epoch(self, epoch: int):
        """Set current epoch for conditional freezing"""
        self.current_epoch = epoch
        
        if self.freeze_backbone and epoch >= self.freeze_epochs:
            self._unfreeze_backbone()
            self.freeze_backbone = False
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_boundary: bool = True,
        mask_labels: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            pixel_values: Input images [B, 3, H, W]
            labels: Ground truth masks [B, H, W] (optional, for semantic segmentation)
            return_boundary: Whether to compute boundary predictions
            mask_labels: List of mask labels for instance segmentation [B, num_instances, H, W]
            class_labels: List of class labels for each mask [B, num_instances]
        
        Returns:
            Dictionary containing:
                - logits: Segmentation logits [B, num_classes, H, W]
                - boundary_logits: Boundary logits [B, 1, H, W] (if add_boundary_head)
                - masks_queries_logits: Mask predictions from queries [B, num_queries, H, W]
                - class_queries_logits: Class predictions for queries [B, num_queries, num_classes]
        """
        B, C, H, W = pixel_values.shape
        
        # Prepare inputs for Mask2Former
        # Mask2Former expects mask_labels and class_labels for training
        if labels is not None and mask_labels is None:
            # Convert semantic labels to instance format
            mask_labels = []
            class_labels = []
            
            for b in range(B):
                label = labels[b]
                unique_classes = torch.unique(label)
                
                # Create mask for each class
                batch_masks = []
                batch_classes = []
                
                for cls in unique_classes:
                    if cls == 0:  # Skip background
                        continue
                    mask = (label == cls).float()
                    batch_masks.append(mask)
                    batch_classes.append(cls)
                
                # Handle case with no iris
                if len(batch_masks) == 0:
                    # Add dummy mask
                    batch_masks.append(torch.zeros_like(label).float())
                    batch_classes.append(torch.tensor(0, device=label.device))
                
                mask_labels.append(torch.stack(batch_masks))
                class_labels.append(torch.tensor(batch_classes, device=label.device))
        
        # Forward through Mask2Former
        outputs = self.mask2former(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels
        )
        
        # Extract predictions
        # Mask2Former outputs: masks_queries_logits and class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits  # [B, num_queries, H, W]
        class_queries_logits = outputs.class_queries_logits  # [B, num_queries, num_classes+1]
        
        # Convert query predictions to semantic segmentation
        # Take argmax over queries for each pixel
        B, Q, Hm, Wm = masks_queries_logits.shape
        
        # Get class probabilities for each query
        class_probs = F.softmax(class_queries_logits, dim=-1)  # [B, Q, num_classes+1]
        
        # Remove "no object" class (last class)
        class_probs = class_probs[..., :-1]  # [B, Q, num_classes]
        
        # Combine mask and class predictions
        # Reshape for broadcasting
        masks_probs = torch.sigmoid(masks_queries_logits)  # [B, Q, H, W]
        
        # Compute semantic segmentation logits
        # For each pixel, compute score for each class
        semantic_logits = torch.zeros(B, self.mask2former.config.num_labels, Hm, Wm, 
                                     device=pixel_values.device)
        
        for b in range(B):
            for cls in range(self.mask2former.config.num_labels):
                # Sum over all queries that predict this class
                class_mask = masks_probs[b] * class_probs[b, :, cls].view(-1, 1, 1)
                semantic_logits[b, cls] = class_mask.sum(dim=0)
        
        # Upsample to input size
        semantic_logits = F.interpolate(
            semantic_logits,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        
        result = {
            'logits': semantic_logits,
            'masks_queries_logits': masks_queries_logits,
            'class_queries_logits': class_queries_logits,
            'loss': outputs.loss if hasattr(outputs, 'loss') else None
        }
        
        # Add auxiliary losses if available
        if self.use_auxiliary_loss and hasattr(outputs, 'auxiliary_logits'):
            result['auxiliary_outputs'] = outputs.auxiliary_logits
        
        # Add boundary prediction
        if self.add_boundary_head and return_boundary:
            boundary_logits = self.boundary_head(semantic_logits)
            result['boundary_logits'] = boundary_logits
        
        return result


class Mask2FormerWithDeepSupervision(EnhancedMask2Former):
    """
    Mask2Former with enhanced deep supervision
    Note: Mask2Former already has built-in auxiliary losses
    """
    
    def __init__(
        self,
        model_name: str = "facebook/mask2former-swin-small-coco-panoptic",
        num_labels: int = 2,
        add_boundary_head: bool = True,
        num_queries: int = 50,
        deep_supervision_weight: float = 0.3
    ):
        super().__init__(
            model_name=model_name,
            num_labels=num_labels,
            add_boundary_head=add_boundary_head,
            use_auxiliary_loss=True
        )
        
        self.deep_supervision_weight = deep_supervision_weight
        
        # Update config for number of queries
        if hasattr(self.mask2former.config, 'num_queries'):
            self.mask2former.config.num_queries = num_queries


def create_mask2former_model(
    model_name: str = "facebook/mask2former-swin-small-coco-panoptic",
    num_labels: int = 2,
    model_type: str = "enhanced",  # "enhanced" or "deep_supervision"
    **kwargs
) -> nn.Module:
    """
    Factory function to create Mask2Former models
    
    Args:
        model_name: HuggingFace model name/path
        num_labels: Number of segmentation classes
        model_type: Type of model enhancement
        **kwargs: Additional model arguments
    
    Returns:
        Model instance
    """
    
    if model_type == "enhanced":
        model = EnhancedMask2Former(
            model_name=model_name,
            num_labels=num_labels,
            **kwargs
        )
    elif model_type == "deep_supervision":
        model = Mask2FormerWithDeepSupervision(
            model_name=model_name,
            num_labels=num_labels,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model


def load_pretrained_mask2former(checkpoint_path: str, model_config: Dict[str, Any]) -> nn.Module:
    """
    Load a pretrained Mask2Former model
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_config: Model configuration dictionary
    
    Returns:
        Loaded model
    """
    model = create_mask2former_model(**model_config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
    
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing Enhanced Mask2Former...")
    model = create_mask2former_model(
        model_type="enhanced",
        add_boundary_head=True,
        freeze_backbone=True,
        freeze_epochs=10
    )
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 512, 512)
    dummy_labels = torch.randint(0, 2, (batch_size, 512, 512))
    
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_input, dummy_labels)
        
        print(f"\nOutput shapes:")
        print(f"Logits: {outputs['logits'].shape}")
        print(f"Masks queries: {outputs['masks_queries_logits'].shape}")
        print(f"Class queries: {outputs['class_queries_logits'].shape}")
        if 'boundary_logits' in outputs:
            print(f"Boundary logits: {outputs['boundary_logits'].shape}")
    
    print("\nModel creation successful!")
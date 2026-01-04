"""
Mask2Former model FIXED for iris segmentation
Key fixes:
1. Proper semantic segmentation conversion
2. Correct instance format handling
3. Post-processing logic aligned with Mask2Former design
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig
from typing import Optional, Dict, Any, List

from .heads import BoundaryRefinementHead


class EnhancedMask2Former(nn.Module):
    """
    Mask2Former with boundary refinement for iris segmentation
    FIXED: Proper semantic segmentation handling
    """
    
    def __init__(
        self,
        model_name: str = "facebook/mask2former-swin-small-coco-panoptic",
        num_labels: int = 2,
        add_boundary_head: bool = True,
        freeze_backbone: bool = False,
        freeze_epochs: int = 0,
        use_auxiliary_loss: bool = True,
        num_queries: int = 50
    ):
        super().__init__()
        
        # Load config first to modify it
        config = Mask2FormerConfig.from_pretrained(model_name)
        config.num_labels = num_labels
        config.num_queries = num_queries
        
        # Load pretrained Mask2Former with modified config
        self.mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True
        )
        
        self.num_labels = num_labels
        self.num_queries = num_queries
        
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
        for param in self.mask2former.model.pixel_level_module.parameters():
            param.requires_grad = False
        print("✅ Mask2Former backbone frozen")
    
    def _unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.mask2former.model.pixel_level_module.parameters():
            param.requires_grad = True
        print("✅ Mask2Former backbone unfrozen")
    
    def set_epoch(self, epoch: int):
        """Set current epoch for conditional freezing"""
        self.current_epoch = epoch
        
        if self.freeze_backbone and epoch >= self.freeze_epochs:
            self._unfreeze_backbone()
            self.freeze_backbone = False
    
    def _prepare_instance_labels(
        self, 
        semantic_labels: torch.Tensor
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        FIXED: Convert semantic labels to instance format properly
        
        Args:
            semantic_labels: [B, H, W] with values {0: background/pupil, 1: iris}
        
        Returns:
            mask_labels: List of [num_instances, H, W] tensors
            class_labels: List of [num_instances] tensors
        """
        B, H, W = semantic_labels.shape
        
        mask_labels = []
        class_labels = []
        
        for b in range(B):
            label = semantic_labels[b]  # [H, W]
            
            # For iris segmentation, we treat each connected component as an instance
            # But typically there's only 1 iris region per image
            
            batch_masks = []
            batch_classes = []
            
            # Create binary mask for iris class
            iris_mask = (label == 1).float()  # [H, W]
            
            if iris_mask.sum() > 0:
                # Add iris instance
                batch_masks.append(iris_mask)
                batch_classes.append(torch.tensor(1, device=label.device))
            
            # Also add background as an instance (Mask2Former expects this)
            background_mask = (label == 0).float()  # [H, W]
            if background_mask.sum() > 0:
                batch_masks.append(background_mask)
                batch_classes.append(torch.tensor(0, device=label.device))
            
            # If no valid instances, add dummy
            if len(batch_masks) == 0:
                batch_masks.append(torch.zeros_like(label).float())
                batch_classes.append(torch.tensor(0, device=label.device))
            
            # Stack masks: [num_instances, H, W]
            mask_labels.append(torch.stack(batch_masks))
            class_labels.append(torch.stack(batch_classes))
        
        return mask_labels, class_labels
    
    def _convert_to_semantic(
        self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: torch.Tensor,
        target_size: tuple
    ) -> torch.Tensor:
        """
        FIXED: Properly convert Mask2Former outputs to semantic segmentation
        
        Args:
            masks_queries_logits: [B, Q, H, W] - mask predictions for each query
            class_queries_logits: [B, Q, num_labels+1] - class predictions (last is "no object")
            target_size: (H, W) target size
        
        Returns:
            semantic_logits: [B, num_labels, H, W]
        """
        B, Q, Hm, Wm = masks_queries_logits.shape
        H, W = target_size
        
        # Get class probabilities (remove "no object" class)
        class_probs = F.softmax(class_queries_logits, dim=-1)  # [B, Q, num_labels+1]
        class_probs = class_probs[..., :-1]  # [B, Q, num_labels] - remove "no object"
        
        # Get mask probabilities
        masks_probs = torch.sigmoid(masks_queries_logits)  # [B, Q, Hm, Wm]
        
        # Upsample masks to target size first
        if (Hm, Wm) != (H, W):
            masks_probs = F.interpolate(
                masks_probs,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )  # [B, Q, H, W]
        
        # Method: For each pixel, find the query with highest score for each class
        # Then assign that class's probability weighted by mask probability
        
        semantic_logits = torch.zeros(B, self.num_labels, H, W, 
                                     device=masks_queries_logits.device)
        
        for b in range(B):
            for cls in range(self.num_labels):
                # Get class probability for this class across all queries: [Q]
                cls_prob = class_probs[b, :, cls]  # [Q]
                
                # Get mask probabilities for all queries: [Q, H, W]
                mask_prob = masks_probs[b]  # [Q, H, W]
                
                # Combine: each query contributes to this class based on its class prob
                # Weighted combination: [Q, H, W] * [Q, 1, 1] -> [Q, H, W]
                weighted_masks = mask_prob * cls_prob.view(-1, 1, 1)
                
                # Take max across queries (not sum - avoid over-saturation)
                # This gives us the strongest prediction for this class
                semantic_logits[b, cls] = weighted_masks.max(dim=0)[0]
        
        return semantic_logits
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_boundary: bool = True,
        mask_labels: Optional[List[torch.Tensor]] = None,
        class_labels: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        FIXED Forward pass with proper instance handling
        
        Args:
            pixel_values: [B, 3, H, W]
            labels: [B, H, W] semantic labels (will be converted to instance format)
            return_boundary: Whether to compute boundary predictions
            mask_labels: Optional pre-formatted instance masks
            class_labels: Optional pre-formatted instance classes
        
        Returns:
            Dictionary containing logits, masks, classes, and optionally boundary
        """
        B, C, H, W = pixel_values.shape
        
        # Prepare instance format if semantic labels provided
        if labels is not None and mask_labels is None:
            mask_labels, class_labels = self._prepare_instance_labels(labels)
        
        # Forward through Mask2Former
        outputs = self.mask2former(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels
        )
        
        # Extract predictions
        masks_queries_logits = outputs.masks_queries_logits  # [B, Q, Hm, Wm]
        class_queries_logits = outputs.class_queries_logits  # [B, Q, num_labels+1]
        
        # Convert to semantic segmentation
        semantic_logits = self._convert_to_semantic(
            masks_queries_logits,
            class_queries_logits,
            target_size=(H, W)
        )
        
        result = {
            'logits': semantic_logits,  # [B, num_labels, H, W]
            'masks_queries_logits': masks_queries_logits,
            'class_queries_logits': class_queries_logits,
            'loss': outputs.loss if hasattr(outputs, 'loss') else None
        }
        
        # Add auxiliary outputs if available
        if self.use_auxiliary_loss and hasattr(outputs, 'auxiliary_predictions'):
            result['auxiliary_outputs'] = outputs.auxiliary_predictions
        
        # Add boundary prediction
        if self.add_boundary_head and return_boundary:
            boundary_logits = self.boundary_head(semantic_logits)
            result['boundary_logits'] = boundary_logits
        
        return result


def create_model(
    architecture: str = "mask2former",
    model_name: str = "facebook/mask2former-swin-small-coco-panoptic",
    num_labels: int = 2,
    model_type: str = "enhanced",
    **kwargs
) -> nn.Module:
    """
    FIXED: Factory function to create models
    """
    if architecture != "mask2former":
        raise ValueError(f"This file only supports mask2former, got {architecture}")
    
    if model_type == "enhanced":
        model = EnhancedMask2Former(
            model_name=model_name,
            num_labels=num_labels,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    print("Testing FIXED Mask2Former model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_model(
        architecture="mask2former",
        model_type="enhanced",
        add_boundary_head=True,
        num_queries=50
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
        
        print(f"\n✅ Output shapes:")
        print(f"   Semantic logits: {outputs['logits'].shape}")
        print(f"   Masks queries: {outputs['masks_queries_logits'].shape}")
        print(f"   Class queries: {outputs['class_queries_logits'].shape}")
        if 'boundary_logits' in outputs:
            print(f"   Boundary logits: {outputs['boundary_logits'].shape}")
    
    print("\n✅ Model test passed!")
"""
Mask2Former model FIXED for iris segmentation
Key fixes:
1. Added **kwargs to handle extra config parameters (architecture, model_type, etc.)
2. Proper semantic segmentation conversion
3. Robust instance format handling
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
    """
    
    def __init__(
        self,
        model_name: str = "facebook/mask2former-swin-small-coco-panoptic",
        num_labels: int = 2,
        add_boundary_head: bool = True,
        freeze_backbone: bool = False,
        freeze_epochs: int = 0,
        use_auxiliary_loss: bool = True,
        num_queries: int = 50,
        **kwargs  # <--- FIX QUAN TRỌNG: Chấp nhận các tham số thừa (như architecture, model_type)
    ):
        super().__init__()
        
        # In ra các tham số thừa để debug (tùy chọn)
        # if kwargs:
        #     print(f"⚠️ EnhancedMask2Former ignored extra args: {list(kwargs.keys())}")

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
        Convert semantic labels to instance format properly
        """
        B, H, W = semantic_labels.shape
        
        mask_labels = []
        class_labels = []
        
        for b in range(B):
            label = semantic_labels[b]
            batch_masks = []
            batch_classes = []
            
            # Iris Mask (Class 1)
            iris_mask = (label == 1).float()
            if iris_mask.sum() > 0:
                batch_masks.append(iris_mask)
                batch_classes.append(torch.tensor(1, device=label.device))
            
            # Background Mask (Class 0)
            background_mask = (label == 0).float()
            if background_mask.sum() > 0:
                batch_masks.append(background_mask)
                batch_classes.append(torch.tensor(0, device=label.device))
            
            # Safety check: if empty, add dummy
            if len(batch_masks) == 0:
                batch_masks.append(torch.zeros_like(label).float())
                batch_classes.append(torch.tensor(0, device=label.device))
            
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
        Convert Mask2Former outputs to semantic segmentation
        """
        B, Q, Hm, Wm = masks_queries_logits.shape
        H, W = target_size
        
        # Get class probabilities (remove "no object" class)
        class_probs = F.softmax(class_queries_logits, dim=-1)  # [B, Q, num_labels+1]
        class_probs = class_probs[..., :-1]  # [B, Q, num_labels]
        
        # Get mask probabilities
        masks_probs = torch.sigmoid(masks_queries_logits)
        
        # Upsample masks if needed
        if (Hm, Wm) != (H, W):
            masks_probs = F.interpolate(
                masks_probs,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
        
        semantic_logits = torch.zeros(B, self.num_labels, H, W, 
                                     device=masks_queries_logits.device)
        
        for b in range(B):
            for cls in range(self.num_labels):
                cls_prob = class_probs[b, :, cls]  # [Q]
                mask_prob = masks_probs[b]  # [Q, H, W]
                
                # Weighted combination
                weighted_masks = mask_prob * cls_prob.view(-1, 1, 1)
                
                # Take max across queries
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
        
        B, C, H, W = pixel_values.shape
        
        if labels is not None and mask_labels is None:
            mask_labels, class_labels = self._prepare_instance_labels(labels)
        
        outputs = self.mask2former(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels
        )
        
        masks_queries_logits = outputs.masks_queries_logits
        class_queries_logits = outputs.class_queries_logits
        
        semantic_logits = self._convert_to_semantic(
            masks_queries_logits,
            class_queries_logits,
            target_size=(H, W)
        )
        
        result = {
            'logits': semantic_logits,
            'masks_queries_logits': masks_queries_logits,
            'class_queries_logits': class_queries_logits,
            'loss': outputs.loss if hasattr(outputs, 'loss') else None
        }
        
        if self.use_auxiliary_loss and hasattr(outputs, 'auxiliary_predictions'):
            result['auxiliary_outputs'] = outputs.auxiliary_predictions
        
        if self.add_boundary_head and return_boundary:
            boundary_logits = self.boundary_head(semantic_logits)
            result['boundary_logits'] = boundary_logits
        
        return result


def create_model(
    architecture: str = "mask2former",
    model_name: str = "facebook/mask2former-swin-small-coco-panoptic",
    num_labels: int = 2,
    model_type: str = "enhanced",
    **kwargs # <--- FIX: Chấp nhận mọi tham số khác
) -> nn.Module:
    """
    Factory function to create models
    """
    if architecture != "mask2former":
        raise ValueError(f"This file only supports mask2former, got {architecture}")
    
    if model_type == "enhanced":
        # Truyền kwargs vào class
        model = EnhancedMask2Former(
            model_name=model_name,
            num_labels=num_labels,
            **kwargs 
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model
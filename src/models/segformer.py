"""
Enhanced SegFormer implementations for iris segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from typing import Optional, Dict, Any

from .heads import BoundaryRefinementHead, AuxiliaryHead


class EnhancedSegFormer(nn.Module):
    """
    SegFormer with boundary refinement head for iris segmentation
    """
    
    def __init__(
        self,
        model_name: str = "nvidia/segformer-b1-finetuned-ade-512-512",
        num_labels: int = 2,
        add_boundary_head: bool = True,
        freeze_encoder: bool = False,
        freeze_epochs: int = 0
    ):
        super().__init__()
        
        # Load pretrained SegFormer
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
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
        self.freeze_encoder = freeze_encoder
        self.freeze_epochs = freeze_epochs
        self.current_epoch = 0
        
        if freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        """Freeze encoder parameters"""
        for param in self.segformer.segformer.encoder.parameters():
            param.requires_grad = False
        print("SegFormer encoder frozen")
    
    def _unfreeze_encoder(self):
        """Unfreeze encoder parameters"""
        for param in self.segformer.segformer.encoder.parameters():
            param.requires_grad = True
        print("SegFormer encoder unfrozen")
    
    def set_epoch(self, epoch: int):
        """Set current epoch for conditional freezing"""
        self.current_epoch = epoch
        
        if self.freeze_encoder and epoch >= self.freeze_epochs:
            self._unfreeze_encoder()
            self.freeze_encoder = False  # Don't unfreeze again
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_boundary: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            pixel_values: Input images [B, 3, H, W]
            labels: Ground truth masks [B, H, W] (optional)
            return_boundary: Whether to compute boundary predictions
        
        Returns:
            Dictionary containing:
                - logits: Segmentation logits [B, num_classes, H, W]
                - boundary_logits: Boundary logits [B, 1, H, W] (if add_boundary_head)
                - loss: Combined loss (if labels provided)
        """
        # Get SegFormer outputs
        outputs = self.segformer(pixel_values=pixel_values, labels=labels)
        
        # Upsample logits to input size
        upsampled_logits = F.interpolate(
            outputs.logits,
            size=pixel_values.shape[-2:],  # (H, W)
            mode='bilinear',
            align_corners=False
        )
        
        result = {
            'logits': upsampled_logits,
            'seg_loss': outputs.loss if labels is not None else None
        }
        
        # Add boundary prediction
        if self.add_boundary_head and return_boundary:
            # Use segmentation logits as input to boundary head
            boundary_logits = self.boundary_head(upsampled_logits)
            result['boundary_logits'] = boundary_logits
        
        return result


class DeepSupervisionSegFormer(EnhancedSegFormer):
    """
    SegFormer with deep supervision for better training
    """
    
    def __init__(
        self,
        model_name: str = "nvidia/segformer-b1-finetuned-ade-512-512",
        num_labels: int = 2,
        add_boundary_head: bool = True,
        deep_supervision: bool = True,
        aux_loss_weight: float = 0.2
    ):
        super().__init__(model_name, num_labels, add_boundary_head)
        
        self.deep_supervision = deep_supervision
        self.aux_loss_weight = aux_loss_weight
        
        if deep_supervision:
            # Add auxiliary classifier from intermediate features
            # SegFormer-B1 has hidden_sizes = [64, 128, 320, 512]
            # We'll tap the 3rd stage (320 channels)
            self.aux_classifier = AuxiliaryHead(
                in_channels=320,
                num_classes=num_labels,
                dropout=0.1
            )
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_boundary: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward with deep supervision"""
        
        # Get encoder features
        encoder_outputs = self.segformer.segformer.encoder(pixel_values)
        hidden_states = encoder_outputs.last_hidden_states
        
        # Get main segmentation outputs
        outputs = self.segformer(pixel_values=pixel_values, labels=labels)
        
        # Upsample logits to input size
        upsampled_logits = F.interpolate(
            outputs.logits,
            size=pixel_values.shape[-2:],  # (H, W)
            mode='bilinear',
            align_corners=False
        )
        
        result = {
            'logits': upsampled_logits,
            'seg_loss': outputs.loss if labels is not None else None
        }
        
        # Deep supervision from 3rd stage
        if self.deep_supervision and labels is not None:
            # hidden_states[2] is from 3rd stage
            aux_logits = self.aux_classifier(hidden_states[2])
            
            # Upsample auxiliary logits to match label size
            aux_logits = F.interpolate(
                aux_logits,
                size=labels.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            
            # Compute auxiliary loss
            aux_loss = F.cross_entropy(aux_logits, labels)
            result['aux_logits'] = aux_logits
            result['aux_loss'] = aux_loss
        
        # Add boundary prediction
        if self.add_boundary_head and return_boundary:
            boundary_logits = self.boundary_head(upsampled_logits)
            result['boundary_logits'] = boundary_logits
        
        return result


def create_model(
    model_name: str = "nvidia/segformer-b1-finetuned-ade-512-512",
    num_labels: int = 2,
    model_type: str = "enhanced",  # "enhanced" or "deep_supervision"
    **kwargs
) -> nn.Module:
    """
    Factory function to create SegFormer models
    
    Args:
        model_name: HuggingFace model name/path
        num_labels: Number of segmentation classes
        model_type: Type of model enhancement
        **kwargs: Additional model arguments
    
    Returns:
        Model instance
    """
    
    if model_type == "enhanced":
        model = EnhancedSegFormer(
            model_name=model_name,
            num_labels=num_labels,
            **kwargs
        )
    elif model_type == "deep_supervision":
        model = DeepSupervisionSegFormer(
            model_name=model_name,
            num_labels=num_labels,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model


def load_pretrained_iris_model(checkpoint_path: str, model_config: Dict[str, Any]) -> nn.Module:
    """
    Load a pretrained iris segmentation model
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_config: Model configuration dictionary
    
    Returns:
        Loaded model
    """
    model = create_model(**model_config)
    
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

"""
Neural network heads for enhanced SegFormer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryRefinementHead(nn.Module):
    """
    Lightweight boundary refinement head for sharp boundary prediction
    """
    
    def __init__(self, in_channels: int = 2, hidden_channels: int = 64):
        super().__init__()
        
        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1, bias=True)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, seg_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seg_logits: Segmentation logits [B, num_classes, H, W]
        
        Returns:
            Boundary logits [B, 1, H, W]
        """
        boundary_logits = self.refine_conv(seg_logits)
        return boundary_logits


class AuxiliaryHead(nn.Module):
    """
    Auxiliary classification head for deep supervision
    """
    
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, num_classes, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Feature maps [B, C, H, W]
        
        Returns:
            Classification logits [B, num_classes, H, W]
        """
        return self.classifier(features)


class AttentionRefinementHead(nn.Module):
    """
    Attention-based refinement head for improved segmentation
    """
    
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1, bias=True),
            nn.Sigmoid()
        )
        
        self.refiner = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Feature maps [B, C, H, W]
        
        Returns:
            Refined logits [B, num_classes, H, W]
        """
        attention_weights = self.attention(features)
        refined_features = features * attention_weights
        return self.refiner(refined_features)

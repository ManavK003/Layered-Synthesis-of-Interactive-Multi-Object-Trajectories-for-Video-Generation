"""
Layered Multi-Object Attention for Video Generation
Novel contribution: Hierarchical attention layers for multi-object control
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import numpy as np


class ObjectLayer(nn.Module):
    """Single layer in hierarchical attention stack"""
    
    def __init__(self, layer_idx: int, hidden_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention for this layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Cross-attention between objects in this layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1)
        )
        
    def forward(
        self, 
        latents: torch.Tensor, 
        object_mask: torch.Tensor,
        other_objects_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply masked attention for this layer's objects
        
        Args:
            latents: [B, H*W*T, C] flattened latent representation
            object_mask: [B, H*W*T] binary mask for current layer objects
            other_objects_context: [B, H*W*T, C] context from other layers
            
        Returns:
            attended_latents: [B, H*W*T, C] attended features
        """
        batch_size = latents.shape[0]
        
        # Expand mask for attention
        mask_expanded = object_mask.unsqueeze(-1)  # [B, H*W*T, 1]
        
        # Apply mask to latents
        masked_latents = latents * mask_expanded
        
        # Self-attention within this layer's objects
        attn_out, _ = self.self_attention(
            masked_latents, 
            masked_latents, 
            masked_latents,
            key_padding_mask=~(object_mask.bool())  # Mask out non-object regions
        )
        latents = self.norm1(latents + attn_out)
        
        # Cross-attention with other layers (if provided)
        if other_objects_context is not None:
            cross_attn_out, _ = self.cross_attention(
                latents,
                other_objects_context,
                other_objects_context
            )
            latents = self.norm2(latents + cross_attn_out)
        
        # Feed-forward network
        ffn_out = self.ffn(latents)
        latents = self.norm3(latents + ffn_out)
        
        # Apply mask again to ensure only object regions are modified
        latents = latents * mask_expanded + latents * (1 - mask_expanded) * 0.1
        
        return latents


class LayeredMultiObjectAttention(nn.Module):
    """
    Hierarchical Multi-Object Attention Module
    Novel approach: Processes objects in hierarchical layers with inter-layer communication
    """
    
    def __init__(
        self, 
        num_layers: int = 4,
        hidden_dim: int = 512,
        num_heads: int = 8,
        enable_cross_layer: bool = True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.enable_cross_layer = enable_cross_layer
        
        # Create hierarchical layers
        self.layers = nn.ModuleList([
            ObjectLayer(i, hidden_dim, num_heads) 
            for i in range(num_layers)
        ])
        
        # Layer priority encoder (learns importance of each layer)
        self.priority_encoder = nn.Sequential(
            nn.Linear(num_layers, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_layers),
            nn.Softmax(dim=-1)
        )
        
    def get_objects_for_layer(
        self, 
        object_masks: List[torch.Tensor], 
        object_priorities: List[int],
        layer_idx: int
    ) -> torch.Tensor:
        """
        Assign objects to appropriate layer based on priority
        
        Args:
            object_masks: List of [B, H, W, T] masks for each object
            object_priorities: List of priority values (0=highest priority)
            layer_idx: Current layer index
            
        Returns:
            layer_mask: [B, H*W*T] combined mask for this layer
        """
        layer_mask = None
        
        for obj_idx, priority in enumerate(object_priorities):
            if priority == layer_idx:
                obj_mask = object_masks[obj_idx]
                # Flatten spatial and temporal dimensions
                obj_mask_flat = obj_mask.flatten(1, 3)  # [B, H*W*T]
                
                if layer_mask is None:
                    layer_mask = obj_mask_flat
                else:
                    layer_mask = torch.maximum(layer_mask, obj_mask_flat)
        
        if layer_mask is None:
            # Empty layer, return zeros
            batch_size = object_masks[0].shape[0]
            seq_len = object_masks[0][0].numel()
            layer_mask = torch.zeros(batch_size, seq_len).to(object_masks[0].device)
        
        return layer_mask
    
    def forward(
        self,
        latents: torch.Tensor,
        object_masks: List[torch.Tensor],
        object_priorities: List[int]
    ) -> torch.Tensor:
        """
        Process latents through hierarchical layers
        
        Args:
            latents: [B, C, H, W, T] latent features from diffusion model
            object_masks: List of [B, 1, H, W, T] masks for each object
            object_priorities: List of priority values for layer assignment
            
        Returns:
            processed_latents: [B, C, H, W, T] attended latents
        """
        B, C, H, W, T = latents.shape
        
        # Reshape latents for transformer: [B, H*W*T, C]
        latents_flat = latents.permute(0, 2, 3, 4, 1).reshape(B, H*W*T, C)
        
        # Store outputs from each layer for cross-layer attention
        layer_outputs = []
        
        # Process each layer sequentially
        for layer_idx, layer in enumerate(self.layers):
            # Get mask for current layer
            layer_mask = self.get_objects_for_layer(
                object_masks, 
                object_priorities, 
                layer_idx
            )
            
            # Get context from previous layers (if cross-layer enabled)
            other_context = None
            if self.enable_cross_layer and len(layer_outputs) > 0:
                # Aggregate all previous layer outputs
                other_context = torch.stack(layer_outputs, dim=0).mean(dim=0)
            
            # Apply layer attention
            latents_flat = layer(latents_flat, layer_mask, other_context)
            layer_outputs.append(latents_flat.clone())
        
        # Reshape back to original shape: [B, C, H, W, T]
        processed_latents = latents_flat.reshape(B, H, W, T, C).permute(0, 4, 1, 2, 3)
        
        return processed_latents


class PeekabooLayeredAttention(nn.Module):
    """
    Integration with Peekaboo's masked attention
    Extends Peekaboo to support multi-object hierarchical control
    """
    
    def __init__(
        self,
        unet,  # The 3D UNet from Peekaboo
        num_layers: int = 4,
        hidden_dim: int = 512
    ):
        super().__init__()
        self.unet = unet
        self.layered_attention = LayeredMultiObjectAttention(
            num_layers=num_layers,
            hidden_dim=hidden_dim
        )
        
    def modify_attention_masks(
        self,
        attention_type: str,  # 'spatial', 'cross', or 'temporal'
        original_mask: torch.Tensor,
        object_masks: List[torch.Tensor],
        object_priorities: List[int],
        layer_idx: int
    ) -> torch.Tensor:
        """
        Modify Peekaboo's attention masks for hierarchical multi-object control
        
        Args:
            attention_type: Type of attention being modified
            original_mask: Original Peekaboo mask
            object_masks: List of object masks
            object_priorities: Priority assignments
            layer_idx: Current layer
            
        Returns:
            modified_mask: Hierarchically modified attention mask
        """
        # Get objects for this layer
        layer_mask = self.layered_attention.get_objects_for_layer(
            object_masks, 
            object_priorities, 
            layer_idx
        )
        
        # Combine with original Peekaboo mask
        if attention_type == 'spatial':
            # For spatial attention, use layer-specific masks
            modified_mask = original_mask * layer_mask.unsqueeze(1)
        elif attention_type == 'cross':
            # For cross-attention, blend masks based on priority
            modified_mask = original_mask * (0.5 + 0.5 * layer_mask.unsqueeze(1))
        else:  # temporal
            # For temporal attention, maintain continuity within layers
            modified_mask = original_mask * layer_mask.unsqueeze(1)
        
        return modified_mask
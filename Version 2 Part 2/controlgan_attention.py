"""
ControlGAN-Inspired Attention for RecLMIS Integration

This module implements ControlGAN's sophisticated attention mechanisms:
1. Spatial Attention - establishes word-to-region correspondences
2. Channel Attention - selects important feature channels based on text
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttentionModule(nn.Module):
    """
    ControlGAN spatial attention - establishes word-to-region correspondences.
    Visual features attend to text features to create spatially-aware representations.
    """
    def __init__(self, visual_dim=512, text_dim=512, num_heads=8):
        super(SpatialAttentionModule, self).__init__()
        self.num_heads = num_heads
        self.head_dim = visual_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Multi-head attention projections
        self.W_q = nn.Linear(visual_dim, visual_dim)
        self.W_k = nn.Linear(text_dim, visual_dim)
        self.W_v = nn.Linear(text_dim, visual_dim)
        self.out_proj = nn.Linear(visual_dim, visual_dim)
        
    def forward(self, visual_feat, text_feat, text_mask=None):
        """
        Args:
            visual_feat: (B, C, H, W) or (B, N_v, C)
            text_feat: (B, N_t, C)
            text_mask: (B, N_t) - 1 for valid tokens, 0 for padding
        Returns:
            attended_visual: Same shape as visual_feat
            attention_weights: (B, N_v, N_t)
        """
        # Handle both 4D and 3D inputs
        if len(visual_feat.shape) == 4:
            B, C, H, W = visual_feat.shape
            visual_feat_flat = visual_feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
            is_4d = True
        else:
            B, N_v, C = visual_feat.shape
            visual_feat_flat = visual_feat
            H = W = int(N_v ** 0.5)
            is_4d = False
        
        N_v = visual_feat_flat.shape[1]
        N_t = text_feat.shape[1]
        
        # Project to Q, K, V
        Q = self.W_q(visual_feat_flat)  # (B, N_v, C)
        K = self.W_k(text_feat)          # (B, N_t, C)
        V = self.W_v(text_feat)          # (B, N_t, C)
        
        # Reshape for multi-head attention
        Q = Q.view(B, N_v, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N_t, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N_t, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply text mask if provided
        if text_mask is not None:
            if len(text_mask.shape) == 2:
                mask = text_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N_t)
            else:
                mask = text_mask
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, heads, N_v, N_t)
        
        # Handle NaN from all-masked positions
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)  # (B, heads, N_v, head_dim)
        attended = attended.transpose(1, 2).contiguous().view(B, N_v, C)
        attended = self.out_proj(attended)
        
        # Residual connection
        attended = attended + visual_feat_flat
        
        # Return to original shape if needed
        if is_4d:
            attended = attended.transpose(1, 2).view(B, C, H, W)
        
        # Return mean attention across heads for visualization/weighting
        mean_attn = attn_weights.mean(dim=1)  # (B, N_v, N_t)
        
        return attended, mean_attn


class ChannelAttentionModule(nn.Module):
    """
    ControlGAN channel-wise attention - selects important feature channels
    based on global pooling statistics.
    """
    def __init__(self, channels=512, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            weighted: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Global pooling
        avg_pool = self.avg_pool(x).view(B, C)
        max_pool = self.max_pool(x).view(B, C)
        
        # MLP
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        
        # Channel weights
        weights = self.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        
        return x * weights


class EnhancedInteractor(nn.Module):
    """
    Enhanced Interactor with ControlGAN attention.
    Replaces the original Interactor in RecLMIS.
    
    Returns: (text_feat_out, img_feat_out) - consistent 2-value interface
    """
    def __init__(self, config, vis, img_size=14, channel_num=512, patch_size=1, embed_dim=512):
        super(EnhancedInteractor, self).__init__()
        self.config = config
        self.vis = vis
        self.channel_num = channel_num
        self.embed_dim = embed_dim
        
        # ControlGAN components
        self.spatial_attention = SpatialAttentionModule(
            visual_dim=channel_num,
            text_dim=embed_dim,
            num_heads=8
        )
        
        self.channel_attention = ChannelAttentionModule(
            channels=channel_num,
            reduction=16
        )
        
        # Fusion layer
        self.fusion = nn.Conv2d(channel_num * 2, channel_num, kernel_size=1)
        
        # Text refinement (similar to original Interactor's CTBN)
        self.text_refine = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )
        
        # Store attention weights for reconstruction/visualization
        self.last_attn_weights = None
        self.last_Wpoi = None
        self.last_Wwoi = None
        
    def forward(self, img_feat, text_feat, text_mask=None):
        """
        Args:
            img_feat: (B, C, H, W) - visual features from encoder
            text_feat: (B, N_t, C) - text features from CLIP
            text_mask: (B, N_t) - text mask (1 for valid, 0 for padding)
        Returns:
            text_feat_out: (B, N_t, C) - refined text features
            img_feat_out: (B, N_v, C) - attended image features (flattened)
        """
        B, C, H, W = img_feat.shape
        
        # Apply ControlGAN spatial attention (text guides visual features)
        spatial_attended, attn_weights = self.spatial_attention(
            img_feat, text_feat, text_mask
        )  # spatial_attended: (B, C, H, W), attn_weights: (B, H*W, N_t)
        
        # Store attention weights
        self.last_attn_weights = attn_weights
        
        # Compute importance weights (for compatibility with reconstruction)
        self.last_Wpoi = attn_weights.max(dim=-1).values  # (B, N_v)
        self.last_Wpoi = torch.softmax(self.last_Wpoi, dim=-1)
        self.last_Wwoi = attn_weights.max(dim=1).values   # (B, N_t)
        self.last_Wwoi = torch.softmax(self.last_Wwoi, dim=-1)
        
        # Apply channel attention
        channel_attended = self.channel_attention(img_feat)  # (B, C, H, W)
        
        # Fuse spatial and channel attended features
        combined = torch.cat([spatial_attended, channel_attended], dim=1)  # (B, 2C, H, W)
        fused = self.fusion(combined)  # (B, C, H, W)
        
        # Flatten image features for output
        img_feat_out = fused.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Refine text features (similar to original CTBN)
        text_feat_refined = self.text_refine(text_feat.transpose(1, 2)).transpose(1, 2)
        
        return text_feat_refined, img_feat_out
    
    def get_attention_weights(self):
        """Returns the last computed attention weights."""
        return self.last_attn_weights
    
    def get_importance_weights(self):
        """Returns (Wpoi, Wwoi) for reconstruction compatibility."""
        return self.last_Wpoi, self.last_Wwoi
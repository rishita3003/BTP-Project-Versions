# -*- coding: utf-8 -*-
"""
Original Interactor module - Fixed with consistent interface.
Returns only (text_feat, img_feat) to match EnhancedInteractor.
Attention weights stored as instance attributes for reconstruction.
"""

import torch
import torch.nn as nn
from torch.nn import Dropout, Conv2d
from torch.nn.modules.utils import _pair


class Embeddings(nn.Module):
    """Construct the patch, position embeddings"""
    def __init__(self, config, patch_size, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(0.1)

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class ConvTransBN(nn.Module):
    """(convolution => [BN] => ReLU)"""
    def __init__(self, in_channels, out_channels):
        super(ConvTransBN, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class Interactor(nn.Module):
    """
    Transformer-branch Interactor.
    
    IMPORTANT: Returns (text_feat, img_feat) - only 2 values for consistency
    with EnhancedInteractor. Attention weights stored as attributes.
    """
    def __init__(self, config, vis, img_size, channel_num, patch_size, embed_dim, 
                 depth=1, num_heads=8, mlp_ratio=4., qkv_bias=True, num_classes=1, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super(Interactor, self).__init__()
        self.config = config
        self.embeddings = Embeddings(config=config, patch_size=patch_size, 
                                     img_size=img_size, in_channels=channel_num)
        self.CTBN = ConvTransBN(in_channels=embed_dim, out_channels=embed_dim)
        self.attn_text = nn.MultiheadAttention(embed_dim, num_heads // 2, dropout=0.1)
        self.attn_img = nn.MultiheadAttention(embed_dim, num_heads // 2, dropout=0.1)
        self.scale = embed_dim ** (-0.5)
        
        # Store attention weights for reconstruction (accessed externally)
        self.last_Wpoi = None
        self.last_Wwoi = None
        self.last_attn_weights = None

    def forward(self, img, text_feat, text_mask=None):
        """
        Args:
            img: (B, C, H, W) - visual features
            text_feat: (B, N_t, C) - text features
            text_mask: (B, N_t) - text mask (optional, for compatibility)
        Returns:
            text_feat: (B, N_t, C) - refined text features
            img_feat: (B, N_v, C) - attended image features
            
        Note: NO print statements - weights stored as instance attributes.
        """
        # Embed image features
        img_feat = self.embeddings(img)  # (B, N, C)
        
        # Cross-attention: text attends to image
        text_feat_attn = self.attn_text(
            text_feat.permute(1, 0, 2),
            img_feat.permute(1, 0, 2),
            img_feat.permute(1, 0, 2)
        )[0].permute(1, 0, 2)  # (B, L, C)

        # Cross-attention: image attends to text
        img_feat_attn = self.attn_img(
            img_feat.permute(1, 0, 2),
            text_feat_attn.permute(1, 0, 2),
            text_feat_attn.permute(1, 0, 2)
        )[0].permute(1, 0, 2)  # (B, P, C)

        # Compute attention matrix for weights (stored for reconstruction)
        A = torch.matmul(img_feat_attn, text_feat_attn.transpose(-1, -2)) * self.scale
        A = torch.softmax(A, dim=-1)
        
        # Handle potential NaN from all-masked positions
        A = torch.nan_to_num(A, nan=0.0)

        # Compute and store weights for external access (used by reconstructor)
        Wpoi = A.max(dim=-1).values
        Wwoi = A.max(dim=1).values
        
        self.last_Wpoi = torch.softmax(Wpoi, dim=-1)
        self.last_Wwoi = torch.softmax(Wwoi, dim=-1)
        self.last_attn_weights = A  # (B, N_v, N_t)

        # Refine text features (CTBN = Conv-TransposeBN)
        text_feat_out = text_feat_attn.transpose(1, 2)
        text_feat_out = self.CTBN(text_feat_out)
        text_feat_out = text_feat_out.transpose(1, 2)

        # Return only 2 values to match EnhancedInteractor interface
        return text_feat_out, img_feat_attn
    
    def get_attention_weights(self):
        """Returns the last computed attention weights for reconstruction."""
        return self.last_attn_weights
    
    def get_importance_weights(self):
        """Returns (Wpoi, Wwoi) for reconstruction loss."""
        return self.last_Wpoi, self.last_Wwoi
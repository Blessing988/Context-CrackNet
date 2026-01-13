"""
Model Components for Context-CrackNet

This module contains the building blocks used in Context-CrackNet:
- ConvBlock: Basic convolutional block with BatchNorm and ReLU
- ResNet50Encoder: Pretrained ResNet50 backbone for feature extraction
- AttentionGate: Attention mechanism for skip connections (RFEM)
- LinformerSelfAttention: Efficient self-attention with linear complexity
- LinformerBlock: Transformer block using Linformer attention (CAGM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv -> BatchNorm -> ReLU (two times)
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class ResNet50Encoder(nn.Module):
    """
    ResNet50 encoder for hierarchical feature extraction.
    
    Extracts features at multiple scales:
    - x0: (64, H/2, W/2) - After initial conv
    - x1: (256, H/4, W/4) - After layer1
    - x2: (512, H/8, W/8) - After layer2
    - x3: (1024, H/16, W/16) - After layer3
    - x4: (2048, H/32, W/32) - After layer4
    
    Args:
        pretrained (bool): Whether to use ImageNet pretrained weights
    """
    def __init__(self, pretrained=True):
        super(ResNet50Encoder, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)

        # Initial layers
        self.conv1 = resnet.conv1  # (64, H/2, W/2)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # (64, H/4, W/4)

        # ResNet layers
        self.layer1 = resnet.layer1  # (256, H/4, W/4)
        self.layer2 = resnet.layer2  # (512, H/8, W/8)
        self.layer3 = resnet.layer3  # (1024, H/16, W/16)
        self.layer4 = resnet.layer4  # (2048, H/32, W/32)

    def forward(self, x):
        x = self.conv1(x)  # (B, 64, H/2, W/2)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = x  # (B, 64, H/2, W/2)

        x = self.maxpool(x)  # (B, 64, H/4, W/4)

        x1 = self.layer1(x)  # (B, 256, H/4, W/4)
        x2 = self.layer2(x1)  # (B, 512, H/8, W/8)
        x3 = self.layer3(x2)  # (B, 1024, H/16, W/16)
        x4 = self.layer4(x3)  # (B, 2048, H/32, W/32)

        return x0, x1, x2, x3, x4


class AttentionGate(nn.Module):
    """
    Attention Gate for skip connections (Refined Feature Enhancement Module - RFEM).
    
    Implements attention mechanism that learns to focus on relevant regions
    in the encoder features based on the decoder gating signal.
    
    Args:
        F_g (int): Number of channels in gating signal (from decoder)
        F_l (int): Number of channels in encoder feature map
        F_int (int): Number of intermediate channels
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        # W_g: gating signal (from decoder)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # W_x: encoder feature map
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # Psi: attention coefficient
        self.psi = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, x, g, return_attention=True):
        """
        Args:
            x: Encoder feature map
            g: Gating signal from decoder
            return_attention: Whether to return attention weights
            
        Returns:
            Attended features, optionally with attention weights
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(F.relu(g1 + x1))
        if return_attention:
            return x * psi, psi
        return x * psi


class LinformerSelfAttention(nn.Module):
    """
    Linformer Self-Attention with linear complexity O(n*k).
    
    Projects keys and values to a lower dimension k to reduce the 
    quadratic complexity of standard self-attention.
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        seq_len (int): Sequence length for projection matrices
        k (int): Projection dimension (default: 256)
        dropout (float): Dropout rate
    """
    def __init__(self, embed_dim, num_heads, seq_len=None, k=256, dropout=0.1):
        super(LinformerSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.k = k
        self.seq_len = seq_len
        self.head_dim = embed_dim // num_heads

        # Query, Key, Value linear layers
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Projection matrices for linear attention
        self.proj_E = nn.Parameter(torch.randn(seq_len, k))
        self.proj_F = nn.Parameter(torch.randn(seq_len, k))

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=True):
        batch_size, seq_len, embed_dim = x.size()

        # Linear projections
        Q = self.q_proj(x)  # (batch_size, seq_len, embed_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Project K and V using proj_E and proj_F
        proj_E = self.proj_E.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, k)
        proj_F = self.proj_F.unsqueeze(0).unsqueeze(0)

        K_proj = torch.matmul(K.transpose(-2, -1), proj_E).transpose(-2, -1)
        V_proj = torch.matmul(V.transpose(-2, -1), proj_F).transpose(-2, -1)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K_proj.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, V_proj)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Final linear layer
        attn_output = self.out_proj(attn_output)

        if return_attention:
            return attn_output, attn_probs
        return attn_output
    

class LinformerBlock(nn.Module):
    """
    Transformer block using Linformer attention (Context-Aware Global Module - CAGM).
    
    Combines Linformer self-attention with feed-forward network and 
    residual connections.
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        seq_len (int): Sequence length
        k (int): Linformer projection dimension
        ff_dim (int): Feed-forward network hidden dimension
        dropout (float): Dropout rate
    """
    def __init__(self, embed_dim, num_heads, seq_len, k=256, ff_dim=512, dropout=0.1):
        super(LinformerBlock, self).__init__()
        self.self_attn = LinformerSelfAttention(embed_dim, num_heads, seq_len, k, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, return_attention=True):
        x2 = self.norm1(x)
        if return_attention:
            attn_output, attn_probs = self.self_attn(x2, return_attention=True)
            x = x + self.dropout1(attn_output)
            x2 = self.norm2(x)
            ff_output = self.feed_forward(x2)
            x = x + self.dropout2(ff_output)
            return x, attn_probs
        else:
            attn_output = self.self_attn(x2, return_attention=False)
            x = x + self.dropout1(attn_output)
            x2 = self.norm2(x)
            ff_output = self.feed_forward(x2)
            x = x + self.dropout2(ff_output)
            return x

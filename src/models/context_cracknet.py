"""
Context-CrackNet Model Architecture

This module contains the main Context-CrackNet model and its ablation variant
for crack segmentation in pavement images.

The architecture combines:
- ResNet50 encoder for hierarchical feature extraction
- Linformer-based Context-Aware Global Module (CAGM) for global context
- Attention-gated skip connections (Region Focused Enhancement Module - RFEM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import (
    ConvBlock,
    ResNet50Encoder,
    AttentionGate,
    LinformerBlock,
)


class Context_CrackNet(nn.Module):
    """
    Context-CrackNet: A novel architecture for crack segmentation.
    
    Combines a ResNet50 encoder with Linformer-based global attention (CAGM)
    and attention-gated skip connections (RFEM) for accurate crack detection.
    
    Args:
        in_channels (int): Number of input image channels (default: 3)
        out_channels (int): Number of output segmentation classes (default: 1)
        img_size (int): Input image size, must be divisible by 16 (default: 448)
        num_heads (int): Number of attention heads in Linformer (default: 8)
        ff_dim (int): Feed-forward dimension in Linformer (default: 2048)
        linformer_k (int): Projection dimension for Linformer (default: 256)
        pretrained (bool): Whether to use pretrained ResNet50 weights (default: True)
    
    Example:
        >>> model = Context_CrackNet(in_channels=3, out_channels=1, img_size=448)
        >>> x = torch.randn(1, 3, 448, 448)
        >>> output, attention_maps = model(x)
        >>> output.shape
        torch.Size([1, 1, 448, 448])
    """
    def __init__(self, in_channels=3, out_channels=1, img_size=448, num_heads=8, 
                 ff_dim=2048, linformer_k=256, pretrained=True):
        super(Context_CrackNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder
        self.encoder = ResNet50Encoder(pretrained=pretrained)

        # Linformer Block (CAGM - Context-Aware Global Module)
        seq_len = (img_size // 16) ** 2  # Sequence length at layer3
        self.linformer = LinformerBlock(
            embed_dim=1024, 
            num_heads=num_heads, 
            seq_len=seq_len, 
            k=linformer_k, 
            ff_dim=ff_dim
        )
        
        # Decoder with attention gates (RFEM - Region Focused Enhancement Module)
        self.up4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.attention3 = AttentionGate(F_g=1024, F_l=1024, F_int=512)
        self.conv3 = ConvBlock(2048, 1024)

        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.attention2 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.conv2 = ConvBlock(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.attention1 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.conv1 = ConvBlock(512, 256)

        self.up1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.attention0 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.conv0 = ConvBlock(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        

    def forward(self, x, return_attention=True):
        """
        Forward pass through Context-CrackNet.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            return_attention: Whether to return attention maps
            
        Returns:
            output: Segmentation mask of shape (B, out_channels, H, W)
            attention_maps: List of attention maps if return_attention=True
        """
        # Encoder
        x0, x1, x2, x3, x4 = self.encoder(x)
        # x0: (B, 64, H/2, W/2)
        # x1: (B, 256, H/4, W/4)
        # x2: (B, 512, H/8, W/8)
        # x3: (B, 1024, H/16, W/16)
        # x4: (B, 2048, H/32, W/32)

        # Apply Linformer block to x3 (CAGM)
        b, c, h, w = x3.shape
        x3_flat = x3.view(b, c, -1).permute(0, 2, 1)  # (B, seq_len, 1024)
        
        x3_transformed, linformer_attention = self.linformer(x3_flat, return_attention=True)
        x3 = x3_transformed.permute(0, 2, 1).view(b, c, h, w)

        # Decoder Level 4
        d4 = self.up4(x4)
        x3_att, attn3 = self.attention3(x3, d4, return_attention=True)
        d4 = torch.cat([x3_att, d4], dim=1)
        d4 = self.conv3(d4)

        # Decoder Level 3
        d3 = self.up3(d4)
        x2_att, attn2 = self.attention2(x2, d3, return_attention=True)
        d3 = torch.cat([x2_att, d3], dim=1)
        d3 = self.conv2(d3)

        # Decoder Level 2
        d2 = self.up2(d3)
        x1_att, attn1 = self.attention1(x1, d2, return_attention=True)
        d2 = torch.cat([x1_att, d2], dim=1)
        d2 = self.conv1(d2)

        # Decoder Level 1
        d1 = self.up1(d2)
        x0_att, attn0 = self.attention0(x0, d1, return_attention=True)
        d1 = torch.cat([x0_att, d1], dim=1)
        d1 = self.conv0(d1)

        # Final output
        output = self.final(d1)

        # Upsample to original input size
        output = F.interpolate(output, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        
        if return_attention:
            return output, [attn0, attn1, attn2, attn3, linformer_attention]
        
        return output
    

class Context_CrackNet_ablation(nn.Module):
    """
    Context-CrackNet with optional modules for ablation studies.
    
    Allows selective enabling/disabling of RFEM and CAGM modules
    to study their individual contributions.
    
    Args:
        in_channels (int): Number of input image channels
        out_channels (int): Number of output segmentation classes
        img_size (int): Input image size
        num_heads (int): Number of attention heads in Linformer
        ff_dim (int): Feed-forward dimension in Linformer
        linformer_k (int): Projection dimension for Linformer
        use_rfem (bool): Whether to include the RFEM module (attention gates)
        use_cagm (bool): Whether to include the CAGM module (Linformer)
        pretrained (bool): Whether to use pretrained ResNet50 weights
    """
    def __init__(self, in_channels=3, out_channels=1, img_size=448, num_heads=8, 
                 ff_dim=2048, linformer_k=256, use_rfem=True, use_cagm=True, pretrained=True):
        super(Context_CrackNet_ablation, self).__init__()
        self.use_rfem = use_rfem
        self.use_cagm = use_cagm

        self.encoder = ResNet50Encoder(pretrained=pretrained)

        # Linformer Block (optional based on use_cagm)
        seq_len = (img_size // 16) ** 2
        self.linformer = LinformerBlock(
            embed_dim=1024, num_heads=num_heads, seq_len=seq_len, 
            k=linformer_k, ff_dim=ff_dim
        ) if use_cagm else None

        # Decoder with optional attention gates
        self.up4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.attention3 = AttentionGate(F_g=1024, F_l=1024, F_int=512) if use_rfem else None
        self.conv3 = ConvBlock(2048 if use_rfem else 1024, 1024)

        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.attention2 = AttentionGate(F_g=512, F_l=512, F_int=256) if use_rfem else None
        self.conv2 = ConvBlock(1024 if use_rfem else 512, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.attention1 = AttentionGate(F_g=256, F_l=256, F_int=128) if use_rfem else None
        self.conv1 = ConvBlock(512 if use_rfem else 256, 256)

        self.up1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.attention0 = AttentionGate(F_g=64, F_l=64, F_int=32) if use_rfem else None
        self.conv0 = ConvBlock(128 if use_rfem else 64, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0, x1, x2, x3, x4 = self.encoder(x)

        # Apply Linformer block if enabled
        if self.use_cagm:
            b, c, h, w = x3.shape
            x3_flat = x3.view(b, c, -1).permute(0, 2, 1)
            x3_transformed, _ = self.linformer(x3_flat)
            x3 = x3_transformed.permute(0, 2, 1).view(b, c, h, w)

        # Decoder Level 4
        d4 = self.up4(x4)
        if self.use_rfem:
            x3_att, _ = self.attention3(x3, d4)
            d4 = torch.cat([x3_att, d4], dim=1)
        d4 = self.conv3(d4)

        # Decoder Level 3
        d3 = self.up3(d4)
        if self.use_rfem:
            x2_att, _ = self.attention2(x2, d3)
            d3 = torch.cat([x2_att, d3], dim=1)
        d3 = self.conv2(d3)

        # Decoder Level 2
        d2 = self.up2(d3)
        if self.use_rfem:
            x1_att, _ = self.attention1(x1, d2)
            d2 = torch.cat([x1_att, d2], dim=1)
        d2 = self.conv1(d2)

        # Decoder Level 1
        d1 = self.up1(d2)
        if self.use_rfem:
            x0_att, _ = self.attention0(x0, d1)
            d1 = torch.cat([x0_att, d1], dim=1)
        d1 = self.conv0(d1)

        # Final output
        output = self.final(d1)
        output = F.interpolate(output, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return output

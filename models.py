import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import math
import torchvision.models as models


### Create models for the different baselines using segmentation models pytorch
def create_model(architecture='Unet', encoder_name='resnet50', in_channels=3, num_classes=1, encoder_weights='imagenet'):
    model = getattr(smp, architecture)(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes
    )
    return model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


### Context_CrackNet

# -------------------------------------------
# 1. Resnet50 Block
# -------------------------------------------

class ConvBlock(nn.Module):
    """Basic convolutional block: Conv -> BatchNorm -> ReLU (two times)"""
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


# -------------------------------------------
# 2. AttentionGate: Implements attention mechanism in skip connections
# -------------------------------------------
class AttentionGate(nn.Module):
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
        # x: encoder feature map
        # g: gating signal from decoder
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(F.relu(g1 + x1))
        if return_attention:
            return x * psi, psi
        return x * psi


# -------------------------------------------
# 3. LinformerSelfAttention: Linformer self-attention mechanism
# -------------------------------------------
class LinformerSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len=None, k=256, dropout=0.1):
        super(LinformerSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.k = k
        self.seq_len = seq_len  # Default sequence length is optional
        self.head_dim = embed_dim // num_heads

        # Query, Key, Value linear layers
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # These will now be created dynamically
        # self.proj_E = None
        # self.proj_F = None
        self.seq_len = seq_len
        self.proj_E = nn.Parameter(torch.randn(seq_len, k))
        self.proj_F = nn.Parameter(torch.randn(seq_len, k))

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=True):
        batch_size, seq_len, embed_dim = x.size()

        # Dynamically create proj_E and proj_F based on the actual seq_len
        # if self.proj_E is None or self.proj_F is None or seq_len != self.seq_len:
        #     self.seq_len = seq_len
        #     self.proj_E = nn.Parameter(torch.randn(seq_len, self.k).to(x.device))
        #     self.proj_F = nn.Parameter(torch.randn(seq_len, self.k).to(x.device))

        # Linear projections
        Q = self.q_proj(x)  # (batch_size, seq_len, embed_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Project K and V using proj_E and proj_F
        proj_E = self.proj_E.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, k)
        proj_F = self.proj_F.unsqueeze(0).unsqueeze(0)

        K_proj = torch.matmul(K.transpose(-2, -1), proj_E).transpose(-2, -1)  # (batch_size, num_heads, k, head_dim)
        V_proj = torch.matmul(V.transpose(-2, -1), proj_F).transpose(-2, -1)  # (batch_size, num_heads, k, head_dim)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K_proj.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch_size, num_heads, seq_len, k)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, V_proj)  # (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Final linear layer
        attn_output = self.out_proj(attn_output)  # (batch_size, seq_len, embed_dim)

        if return_attention:
            return attn_output, attn_probs
        return attn_output
    
# -------------------------------------------
# 4. LinformerBlock: Transformer block using LinformerSelfAttention
# -------------------------------------------
class LinformerBlock(nn.Module):
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

# -------------------------------------------
# 5. Context_CrackNet
# -------------------------------------------
class Context_CrackNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, img_size=448, num_heads=8, ff_dim=2048, linformer_k=256, pretrained=True):
        super(Context_CrackNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = ResNet50Encoder(pretrained=pretrained)

        # Linformer Block
        seq_len = (img_size // 16) ** 2  # Assuming img_size is divisible by 16
        self.linformer = LinformerBlock(embed_dim=1024, num_heads=num_heads, seq_len=seq_len, k=linformer_k, ff_dim=ff_dim)
        # Decoder with attention gates
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
        # Encoder
        x0, x1, x2, x3, x4 = self.encoder(x)
        # x0: (B, 64, H/2, W/2)
        # x1: (B, 256, H/4, W/4)
        # x2: (B, 512, H/8, W/8)
        # x3: (B, 1024, H/16, W/16)
        # x4: (B, 2048, H/32, W/32)


        # Apply Linformer block to x3
        b, c, h, w = x3.shape  # x3 is (B, 1024, H/16, W/16)
        x3_flat = x3.view(b, c, -1).permute(0, 2, 1)  # (B, seq_len, 1024)
        
        # attention
        x3_transformed, linformer_attention = self.linformer(x3_flat, return_attention=True)
        # x3_transformed = self.linformer(x3_flat)
        x3 = x3_transformed.permute(0, 2, 1).view(b, c, h, w)  # (B, 1024, H/16, W/16)

        # Decoder Level 4
        d4 = self.up4(x4)  # Upsample x4 to (B, 1024, H/16, W/16)
        x3_att, attn3 = self.attention3(x3, d4, return_attention=True)
        d4 = torch.cat([x3_att, d4], dim=1)  # (B, 1024+1024, H/16, W/16)
        d4 = self.conv3(d4)  # Reduce channels to 1024

        # Decoder Level 3
        d3 = self.up3(d4)  # (B, 512, H/8, W/8)
        x2_att, attn2 = self.attention2(x2, d3, return_attention=True)
        d3 = torch.cat([x2_att, d3], dim=1)  # (B, 512+512, H/8, W/8)
        d3 = self.conv2(d3)  # Reduce channels to 512

        # Decoder Level 2
        d2 = self.up2(d3)  # (B, 256, H/4, W/4)
        x1_att, attn1 = self.attention1(x1, d2, return_attention=True)
        d2 = torch.cat([x1_att, d2], dim=1)  # (B, 256+256, H/4, W/4)
        d2 = self.conv1(d2)  # Reduce channels to 256

        # Decoder Level 1
        d1 = self.up1(d2)  # (B, 64, H/2, W/2)
        x0_att, attn0 = self.attention0(x0, d1, return_attention=True)
        d1 = torch.cat([x0_att, d1], dim=1)  # (B, 64+64, H/2, W/2)
        d1 = self.conv0(d1)  # Reduce channels to 64

        # Final output
        output = self.final(d1)  # (B, out_channels, H/2, W/2)

        # Upsample to original input size
        output = F.interpolate(output, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        
        if return_attention:
            return output, [attn0, attn1, attn2, attn3, linformer_attention]
        
        return output
    

# -------------------------------------------
# Ablation experiments
# -------------------------------------------
class Context_CrackNet_ablation(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, img_size=448, num_heads=8, ff_dim=2048, linformer_k=256, 
                 use_rfem=True, use_cagm=True, pretrained=True):
        """
        Initializes the Context_CrackNet model with optional modules for ablation studies.
        
        Args:
            use_rfem (bool): Whether to include the RFEM module.
            use_cagm (bool): Whether to include the CAGM module.
        """
        super(Context_CrackNet_ablation, self).__init__()
        self.use_rfem = use_rfem
        self.use_cagm = use_cagm

        self.encoder = ResNet50Encoder(pretrained=pretrained)

        # Linformer Block
        seq_len = (img_size // 16) ** 2  # Assuming img_size is divisible by 16
        self.linformer = LinformerBlock(embed_dim=1024, num_heads=num_heads, seq_len=seq_len, k=linformer_k, ff_dim=ff_dim) if use_cagm else None

        # Decoder with attention gates
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
            x3_flat = x3.view(b, c, -1).permute(0, 2, 1)  # (B, seq_len, 1024)
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
            x2_att,  _ = self.attention2(x2, d3)
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
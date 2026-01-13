"""
Baseline Model Factory

This module provides a factory function to create baseline segmentation models
using the segmentation-models-pytorch library.

Supported architectures:
- Unet
- UnetPlusPlus
- PSPNet
- PAN
- MAnet
- Linknet
- FPN
- DeepLabV3Plus
- DeepLabV3
"""

import torch
import segmentation_models_pytorch as smp


def create_model(architecture='Unet', encoder_name='resnet50', in_channels=3, 
                 num_classes=1, encoder_weights='imagenet'):
    """
    Create a baseline segmentation model using segmentation-models-pytorch.
    
    Args:
        architecture (str): Model architecture name. One of:
            'Unet', 'UnetPlusPlus', 'PSPNet', 'PAN', 'MAnet', 
            'Linknet', 'FPN', 'DeepLabV3Plus', 'DeepLabV3'
        encoder_name (str): Encoder backbone name (default: 'resnet50')
        in_channels (int): Number of input channels (default: 3)
        num_classes (int): Number of output classes (default: 1)
        encoder_weights (str): Pretrained weights source (default: 'imagenet')
        
    Returns:
        nn.Module: Segmentation model on appropriate device
        
    Example:
        >>> model = create_model('DeepLabV3Plus', encoder_name='resnet50')
        >>> x = torch.randn(1, 3, 448, 448)
        >>> output = model(x)
        >>> output.shape
        torch.Size([1, 1, 448, 448])
    """
    model = getattr(smp, architecture)(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes
    )
    return model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


# List of supported architectures for reference
SUPPORTED_ARCHITECTURES = [
    'Unet', 'UnetPlusPlus', 'PSPNet', 'PAN', 'MAnet', 
    'Linknet', 'FPN', 'DeepLabV3Plus', 'DeepLabV3'
]

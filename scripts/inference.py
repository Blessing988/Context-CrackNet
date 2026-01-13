#!/usr/bin/env python
"""
Inference Script for Context-CrackNet

This script runs inference on trained models to generate segmentation predictions.
Supports both Context-CrackNet and baseline architectures.

Usage:
    python scripts/inference.py --config configs/config.yaml --checkpoint path/to/model.pth

    # Run on specific dataset and architecture
    python scripts/inference.py --dataset DeepCrack --architecture Context_CrackNet

    # Batch inference on multiple datasets/architectures
    python scripts/inference.py --batch
"""

import os
import sys
import argparse

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from src.models import Context_CrackNet, create_model


# Supported datasets and architectures
DATASETS = [
    'CFD', 'DeepCrack', 'CRACK500', 'cracktree200',
    'Eugen_Muller', 'forest', 'GAPS384', 'Rissbilder',
    'Sylvie', 'Volker'
]

ARCHITECTURES = [
    'Context_CrackNet', 'Unet', 'UnetPlusPlus', 'PSPNet',
    'PAN', 'MAnet', 'Linknet', 'FPN', 'DeepLabV3Plus', 'DeepLabV3'
]


def get_val_transforms():
    """Get validation transforms for inference."""
    return A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])


def load_model(architecture, checkpoint_path, num_classes=1, device='cuda'):
    """
    Load a trained model from checkpoint.
    
    Args:
        architecture: Model architecture name
        checkpoint_path: Path to model checkpoint
        num_classes: Number of output classes
        device: Device to load model on
        
    Returns:
        nn.Module: Loaded model in eval mode
    """
    if architecture == 'Context_CrackNet':
        model = Context_CrackNet(
            in_channels=3,
            out_channels=num_classes,
            img_size=448,
            num_heads=8,
            ff_dim=2048,
            linformer_k=256,
            pretrained=False  # Don't need pretrained weights when loading checkpoint
        )
    else:
        model = create_model(
            architecture=architecture,
            encoder_name='resnet50',
            in_channels=3,
            num_classes=num_classes,
            encoder_weights=None
        )
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


def run_inference(model, image_path, transforms, device, threshold=0.5):
    """
    Run inference on a single image.
    
    Args:
        model: Trained model
        image_path: Path to input image
        transforms: Albumentations transforms
        device: Device to run inference on
        threshold: Prediction threshold for binary segmentation
        
    Returns:
        numpy.ndarray: Predicted binary mask
    """
    # Load and preprocess image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Apply transforms
    transformed = transforms(image=image_rgb)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(image_tensor)
        
        # Handle models that return (output, attention_maps)
        if isinstance(output, tuple):
            output, _ = output
    
    # Apply sigmoid and threshold
    pred = torch.sigmoid(output)
    pred_mask = (pred > threshold).cpu().numpy().astype(np.uint8)
    
    # Squeeze batch and channel dimensions
    pred_mask = pred_mask.squeeze(0).squeeze(0)
    
    # Scale to 0-255
    pred_mask = pred_mask * 255
    
    return pred_mask


def inference_single(args):
    """Run inference on a single dataset/architecture combination."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup paths
    image_dir = os.path.join(args.data_root, args.dataset, 'val', 'images')
    checkpoint_path = os.path.join(
        args.checkpoint_root, args.dataset, args.architecture, 'best_attention.pth'
    )
    output_dir = os.path.join(args.output_root, args.dataset, args.architecture)
    
    if not os.path.isdir(image_dir):
        print(f"Image directory does not exist: {image_dir}")
        return
    
    if not os.path.isfile(checkpoint_path):
        print(f"Checkpoint does not exist: {checkpoint_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading {args.architecture} model from {checkpoint_path}")
    model = load_model(args.architecture, checkpoint_path, num_classes=1, device=device)
    
    # Get transforms
    transforms = get_val_transforms()
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    print(f"Found {len(image_files)} images")
    
    # Run inference
    for image_filename in tqdm(image_files, desc=f"Inference on {args.dataset}"):
        image_path = os.path.join(image_dir, image_filename)
        
        try:
            pred_mask = run_inference(model, image_path, transforms, device, args.threshold)
            
            # Save predicted mask
            output_path = os.path.join(
                output_dir,
                os.path.splitext(image_filename)[0] + '.png'
            )
            cv2.imwrite(output_path, pred_mask)
            
        except Exception as e:
            print(f"Error processing {image_filename}: {e}")
    
    print(f"\nResults saved to: {output_dir}")


def inference_batch(args):
    """Run inference on multiple datasets and architectures."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    datasets = args.datasets if args.datasets else DATASETS
    architectures = args.architectures if args.architectures else ARCHITECTURES
    
    transforms = get_val_transforms()
    
    for dataset in tqdm(datasets, desc="Datasets"):
        image_dir = os.path.join(args.data_root, dataset, 'val', 'images')
        
        if not os.path.isdir(image_dir):
            print(f"Skipping {dataset}: directory does not exist")
            continue
        
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
        
        for architecture in tqdm(architectures, desc=f"{dataset} - Models", leave=False):
            checkpoint_path = os.path.join(
                args.checkpoint_root, dataset, architecture, 'best_attention.pth'
            )
            output_dir = os.path.join(args.output_root, dataset, architecture)
            
            if not os.path.isfile(checkpoint_path):
                print(f"Skipping {architecture}: checkpoint not found")
                continue
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Load model
            model = load_model(architecture, checkpoint_path, num_classes=1, device=device)
            
            for image_filename in tqdm(image_files, desc=f"{architecture}", leave=False):
                image_path = os.path.join(image_dir, image_filename)
                
                try:
                    pred_mask = run_inference(model, image_path, transforms, device, args.threshold)
                    
                    output_path = os.path.join(
                        output_dir,
                        os.path.splitext(image_filename)[0] + '.png'
                    )
                    cv2.imwrite(output_path, pred_mask)
                    
                except Exception as e:
                    print(f"Error: {dataset}/{architecture}/{image_filename}: {e}")
    
    print(f"\nBatch inference complete! Results saved to: {args.output_root}")


def main():
    parser = argparse.ArgumentParser(description='Run inference with Context-CrackNet models')
    
    # Mode selection
    parser.add_argument('--batch', action='store_true',
                        help='Run batch inference on multiple datasets/architectures')
    
    # Single inference arguments
    parser.add_argument('--dataset', type=str, default='DeepCrack',
                        help='Dataset name (default: DeepCrack)')
    parser.add_argument('--architecture', type=str, default='Context_CrackNet',
                        help='Model architecture (default: Context_CrackNet)')
    
    # Batch inference arguments
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='List of datasets for batch mode')
    parser.add_argument('--architectures', nargs='+', default=None,
                        help='List of architectures for batch mode')
    
    # Paths
    parser.add_argument('--data_root', type=str, 
                        default='/mmfs1/projects/armstrong.aboah/Pavement-distress-segmentation/Public-Datasets',
                        help='Root directory containing datasets')
    parser.add_argument('--checkpoint_root', type=str,
                        default='/mmfs1/projects/armstrong.aboah/Pavement-distress-segmentation/trained_models_test',
                        help='Root directory containing model checkpoints')
    parser.add_argument('--output_root', type=str,
                        default='./predictions',
                        help='Root directory for saving predictions')
    
    # Inference parameters
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Prediction threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    if args.batch:
        inference_batch(args)
    else:
        inference_single(args)


if __name__ == '__main__':
    main()

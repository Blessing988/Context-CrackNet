#!/usr/bin/env python
"""
Training Script for Context-CrackNet

This script trains either Context-CrackNet or baseline segmentation models
for crack detection. The model and training parameters are configured via
a YAML configuration file.

Usage:
    python scripts/train.py --config configs/config.yaml

Supported architectures:
    - Context_CrackNet (proposed model)
    - Unet, UnetPlusPlus, PSPNet, PAN, MAnet, Linknet, FPN, DeepLabV3Plus, DeepLabV3
"""

import os
import sys
import argparse

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import numpy as np

from src.data import get_transforms, get_dataloader
from src.models import Context_CrackNet, Context_CrackNet_ablation, create_model
from src.metrics import evaluate_metrics, save_metrics
from src.losses import CombinedBinaryLoss, CombinedMulticlassLoss


def train_one_epoch(model, dataloader, criterion, optimizer, device, num_classes=1):
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        num_classes: Number of segmentation classes
        
    Returns:
        tuple: (epoch_loss, metrics_list)
    """
    model.train()
    epoch_loss = 0
    metrics = {'IoU': [], 'Dice': [], 'Precision': [], 'Recall': [], 'F1 Score': []}
    
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        
        # Handle models that return (output, attention_maps)
        if isinstance(outputs, tuple):
            outputs, _ = outputs

        if num_classes == 1:
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, masks.float())
            preds = torch.sigmoid(outputs) > 0.5
        else:
            loss = criterion(outputs, masks)
            preds = torch.argmax(outputs, dim=1)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)

        batch_metrics = evaluate_metrics(preds.cpu(), masks.cpu(), num_classes)
        for key in metrics:
            metrics[key].append(batch_metrics[key])

    epoch_loss /= len(dataloader.dataset)
    avg_metrics = {key: np.nanmean(metrics[key]) for key in metrics}

    return epoch_loss, list(avg_metrics.values())


def validate(model, dataloader, criterion, device, num_classes=1):
    """
    Validate the model.
    
    Args:
        model: PyTorch model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to evaluate on
        num_classes: Number of segmentation classes
        
    Returns:
        tuple: (epoch_loss, metrics_list)
    """
    model.eval()
    epoch_loss = 0
    metrics = {'IoU': [], 'Dice': [], 'Precision': [], 'Recall': [], 'F1 Score': []}

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            
            # Handle models that return (output, attention_maps)
            if isinstance(outputs, tuple):
                outputs, _ = outputs

            if num_classes == 1:
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, masks.float())
                preds = torch.sigmoid(outputs) > 0.5
            else:
                loss = criterion(outputs, masks.long())
                preds = torch.argmax(outputs, dim=1)

            epoch_loss += loss.item() * images.size(0)

            batch_metrics = evaluate_metrics(preds.cpu(), masks.cpu(), num_classes)
            for key in metrics:
                metrics[key].append(batch_metrics[key])

    epoch_loss /= len(dataloader.dataset)
    avg_metrics = {key: np.nanmean(metrics[key]) for key in metrics}

    return epoch_loss, list(avg_metrics.values())


def build_model(config, device):
    """
    Build model based on configuration.
    
    Args:
        config: Configuration dictionary
        device: Device to place model on
        
    Returns:
        nn.Module: Initialized model
    """
    architecture = config['model']['architecture']
    num_classes = config['model']['num_classes']
    pretrained = config['model'].get('pretrained', True)
    
    if architecture == 'Context_CrackNet':
        model = Context_CrackNet(
            in_channels=3,
            out_channels=num_classes,
            img_size=448,
            num_heads=8,
            ff_dim=2048,
            linformer_k=256,
            pretrained=pretrained
        )
    elif architecture == 'Context_CrackNet_ablation':
        model = Context_CrackNet_ablation(
            in_channels=3,
            out_channels=num_classes,
            img_size=448,
            num_heads=8,
            ff_dim=2048,
            linformer_k=256,
            use_rfem=config['model'].get('use_rfem', True),
            use_cagm=config['model'].get('use_cagm', True),
            pretrained=pretrained
        )
    else:
        # Use baseline model from segmentation-models-pytorch
        model = create_model(
            architecture=architecture,
            encoder_name=config['model'].get('backbone', 'resnet50'),
            in_channels=3,
            num_classes=num_classes,
            encoder_weights='imagenet' if pretrained else None
        )
    
    return model.to(device)


def main(config_path):
    """
    Main training function.
    
    Args:
        config_path: Path to YAML configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataloaders
    train_transforms, val_transforms = get_transforms()
    
    data_root = config['data']['root_dir']
    dataset_name = config['data']['dataset_name']
    
    train_loader = get_dataloader(
        os.path.join(data_root, dataset_name, 'train', 'images'),
        os.path.join(data_root, dataset_name, 'train', 'masks'),
        config['training']['batch_size'],
        config['data']['num_workers'],
        config['model']['num_classes'],
        transform=train_transforms,
        shuffle=True,
    )

    val_loader = get_dataloader(
        os.path.join(data_root, dataset_name, 'val', 'images'),
        os.path.join(data_root, dataset_name, 'val', 'masks'),
        config['training']['batch_size'],
        config['data']['num_workers'],
        config['model']['num_classes'],
        transform=val_transforms,
        shuffle=False
    )

    # Build model
    model_type = config['model']['architecture']
    model = build_model(config, device)
    print(f"Training {model_type} on {dataset_name}")

    # Setup save paths
    save_dir = config['utils']['save_dir']
    os.makedirs(os.path.join(save_dir, dataset_name, model_type), exist_ok=True)
    SAVE_MODEL_PATH = os.path.join(save_dir, dataset_name, model_type, 'best_attention.pth')
    SAVE_METRIC_PATH = os.path.join(save_dir, dataset_name, model_type, 'metrics_attention.csv')

    # Define loss function
    use_bce = config['model'].get('use_bce', True)
    use_dice = config['model'].get('use_dice', True)
    
    if config['model']['num_classes'] == 1:
        criterion = CombinedBinaryLoss(
            weight_bce=0.5, 
            weight_dice=0.5,
            use_bce=use_bce,
            use_dice=use_dice
        )
    else:
        criterion = CombinedMulticlassLoss(
            num_classes=config['model']['num_classes'], 
            weight_ce=0.5, 
            weight_dice=0.5,
            use_bce=use_bce,
            use_dice=use_dice
        )

    # Define optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Training loop
    num_epochs = config['training']['num_epochs']
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print(f"Starting training for {num_epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 
            config['model']['num_classes']
        )

        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, 
            config['model']['num_classes']
        )

        scheduler.step(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} | Train Metrics: {train_metrics}')
        print(f'Val Loss: {val_loss:.4f} | Val Metrics: {val_metrics}')

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print('✓ Best model saved.')
        else:
            epochs_no_improve += 1
            print(f'No improvement for {epochs_no_improve} epochs.')

        print("-" * 60)

        # Early stopping
        if epochs_no_improve >= config['training']['early_stopping_patience']:
            print('Early stopping triggered!')
            break

        save_metrics(epoch, train_loss, train_metrics, val_loss, val_metrics, SAVE_METRIC_PATH)

    print(f"\nTraining complete! Best model saved to: {SAVE_MODEL_PATH}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Context-CrackNet or baseline models')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    args = parser.parse_args()
    
    main(args.config)

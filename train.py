# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv
from datasets_test import  get_transforms, get_dataloader
from models import create_model, Context_CrackNet, Context_CrackNet_ablation
from metrics import evaluate_metrics, save_metrics
from utils import save_checkpoint
import yaml
import numpy as np
import os
from losses import CombinedBinaryLoss, CombinedMulticlassLoss



def train_one_epoch(model, dataloader, criterion, optimizer, device, num_classes=1):
    model.train()
    epoch_loss = 0
    metrics = {'IoU': [], 'Dice': [], 'Precision': [], 'Recall': [], 'F1 Score': []}
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        if isinstance(outputs, tuple):  # Check if outputs is a tuple
            outputs, _ = outputs  # Extract the predictions (ignore attention maps)

        if num_classes == 1:
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, masks.float()) # This worked
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
    model.eval()
    epoch_loss = 0
    metrics = {'IoU': [], 'Dice': [], 'Precision': [], 'Recall': [], 'F1 Score': []}

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            if isinstance(outputs, tuple):  # Check if outputs is a tuple
                outputs, _ = outputs  # Extract the predictions (ignore attention maps)

            if num_classes == 1:
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, masks.float())
                preds = torch.sigmoid(outputs) > 0.5
            else:
                loss = criterion(outputs, masks.long())
                preds = torch.argmax(outputs, dim=1)

            epoch_loss += loss.item() * images.size(0)

            # Calculate metrics
            batch_metrics = evaluate_metrics(preds.cpu(), masks.cpu(), num_classes)
            for key in metrics:
                metrics[key].append(batch_metrics[key])

    epoch_loss /= len(dataloader.dataset)

    # Average metrics
    avg_metrics = {key: np.nanmean(metrics[key]) for key in metrics}

    return epoch_loss, list(avg_metrics.values())


def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataloaders
    train_transforms, val_transforms = get_transforms()
    train_loader = get_dataloader(
        os.path.join(config['data']['root_dir'], config['data']['dataset_name'], 'train', 'images'),
        os.path.join(config['data']['root_dir'], config['data']['dataset_name'], 'train', 'masks'),
        config['training']['batch_size'],
        config['data']['num_workers'],
        config['model']['num_classes'],
        transform = train_transforms,
        shuffle=True,
    )

    val_loader = get_dataloader(
    os.path.join(config['data']['root_dir'], config['data']['dataset_name'], 'val', 'images'),
    os.path.join(config['data']['root_dir'], config['data']['dataset_name'], 'val', 'masks'),
    config['training']['batch_size'],
    config['data']['num_workers'],
    config['model']['num_classes'],
    transform = val_transforms,
    shuffle=False)

    dataset_name = config['data']['dataset_name']
    model_type = config['model']['architecture']

    save_dir = config['utils']['save_dir']
    os.makedirs(os.path.join(save_dir, dataset_name, model_type), exist_ok=True)
    SAVE_MODEL_PATH = os.path.join(save_dir, dataset_name, model_type, 'best_attention.pth')
    SAVE_METRIC_PATH = os.path.join(save_dir, dataset_name, model_type, 'metrics_attention.csv')

    
    # ContextCrackNet
    model = Context_CrackNet(
        in_channels=3,
        out_channels=config['model']['num_classes'],  # Adjust based on your task
        img_size=448,    # Adjust based on your input image size
        num_heads=8,
        ff_dim=2048,
        linformer_k=256,
        pretrained=True  # Use pretrained ResNet50
    )
    model.to(device)
    
    
    # Ablation experiments
    # model = Context_CrackNet_ablation(
    #     in_channels=3,
    #     out_channels=config['model']['num_classes'],  # Adjust based on your task
    #     img_size=448,    # Adjust based on your input image size
    #     num_heads=8,
    #     ff_dim=2048,
    #     linformer_k=256,
    #     use_rfem=config['model']['use_rfem'],
    #     use_cagm=config['model']['use_cagm'],
    #     pretrained=True  # Use pretrained ResNet50
    # )
    # model.to(device)
    
    
    # Define loss function
    if config['model']['num_classes'] == 1:
        # criterion = nn.BCEWithLogitsLoss()
        criterion = CombinedBinaryLoss(weight_bce=0.5, weight_dice=0.5)
    else:
        # criterion = nn.CrossEntropyLoss()
        criterion = CombinedMulticlassLoss(num_classes=7, weight_ce=0.5, weight_dice=0.5)

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

    print(config['data']['dataset_name'])
    for epoch in range(num_epochs):
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, config['model']['num_classes'])

        # with open(train_csv_path, mode='a', newline='') as train_file:
        #     train_writer = csv.writer(train_file)
        #     train_writer.writerow([epoch + 1, train_loss])

        val_loss, val_metrics = validate(model, val_loader, criterion, device, config['model']['num_classes'])

        # with open(val_csv_path, mode='a', newline='') as val_file:
        #     val_writer = csv.writer(val_file)
        #     val_writer.writerow([
        #         epoch + 1,
        #         val_loss,
        #         val_metrics['IoU'],
        #         val_metrics['Dice'],
        #         val_metrics['Precision'],
        #         val_metrics['Recall'],
        #         val_metrics['F1 Score']
        #     ])

        scheduler.step(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Train Metrics: {train_metrics}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val Metrics: {val_metrics}')
        print()

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            # Save best model
            # torch.save(model.state_dict(), 'best_model.pth')
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print('Best model saved.')
        else:
            epochs_no_improve += 1
            print(f'No improvement for {epochs_no_improve} epochs.')

        # Early stopping
        if epochs_no_improve >= config['training']['early_stopping_patience']:
            print('Early stopping!')
            break

        save_metrics(epoch, train_loss, train_metrics, val_loss, val_metrics, SAVE_METRIC_PATH)


if __name__ == '__main__':
    main()

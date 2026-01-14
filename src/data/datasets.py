"""
Dataset and DataLoader utilities for crack segmentation.

This module provides:
- SegmentationDataset: PyTorch Dataset for loading image-mask pairs
- get_transforms: Albumentations transforms for training and validation
- get_dataloader: Factory function for creating DataLoader instances

Supported datasets:
- CFD, DeepCrack, CRACK500, cracktree200, Eugen_Muller, 
- forest, GAPS384, Rissbilder, Sylvie, Volker
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


# List of supported datasets for reference
SUPPORTED_DATASETS = [
    'CFD', 'DeepCrack', 'CRACK500', 'cracktree200',
    'Eugen_Muller', 'forest', 'GAPS384', 'Rissbilder',
    'Sylvie', 'Volker'
]


def get_transforms():
    """
    Get Albumentations transforms for training and validation.
    
    Training transforms include:
    - Horizontal/vertical flips
    - Random rotation
    - Shift, scale, rotate
    - Gaussian noise
    - Color jitter
    - ImageNet normalization
    
    Validation transforms include:
    - ImageNet normalization only
    
    Returns:
        tuple: (train_transforms, val_transforms)
    """
    train_transforms = A.Compose([
        # Spatial-level transforms (applied to both image and mask)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.1, 
            rotate_limit=30, 
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT
        ),
        
        # Image-only transforms
        A.GaussNoise(p=0.2),
        A.ColorJitter(
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.2, 
            hue=0.1, 
            p=0.5
        ),
        
        # Normalize using ImageNet mean and std
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        ),
        # Convert to PyTorch tensors
        ToTensorV2()
    ],
    additional_targets={"mask": "mask"})

    val_transforms = A.Compose([
        # Normalize using ImageNet mean and std
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        ),
        # Convert to PyTorch tensors
        ToTensorV2()
    ], 
    additional_targets={"mask": "mask"})
    
    return train_transforms, val_transforms


class SegmentationDataset(Dataset):
    """
    PyTorch Dataset for crack segmentation.
    
    Loads image-mask pairs from directories and applies transforms.
    
    Args:
        images_dir (str): Path to directory containing images
        masks_dir (str): Path to directory containing masks
        transform: Albumentations transform to apply
        num_classes (int): Number of segmentation classes (default: 1 for binary)
        
    Expected directory structure:
        images_dir/
            image1.jpg
            image2.jpg
            ...
        masks_dir/
            image1.jpg (or .png)
            image2.jpg (or .png)
            ...
    """
    def __init__(self, images_dir, masks_dir, transform=None, num_classes=1):
        super(SegmentationDataset, self).__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        self.num_classes = num_classes

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(
            self.masks_dir, 
            os.path.splitext(self.image_files[idx])[0] + '.jpg'
        )
            
        # Load image and convert BGR to RGB
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask as grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
        # Preprocess mask to binary
        mask = self.preprocess_mask(mask)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        return image, mask.long()
    
    def preprocess_mask(self, mask):
        """Convert mask to binary format (0 or 1)."""
        mask = np.array(mask)
        mask = (mask > 0).astype(np.uint8)
        return mask

        
def get_dataloader(images_dir, masks_dir, batch_size, num_workers, num_classes, 
                   transform, shuffle):
    """
    Create a DataLoader for the segmentation dataset.
    
    Args:
        images_dir (str): Path to images directory
        masks_dir (str): Path to masks directory
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        num_classes (int): Number of segmentation classes
        transform: Albumentations transform
        shuffle (bool): Whether to shuffle the data
        
    Returns:
        DataLoader: PyTorch DataLoader instance
    """
    dataset = SegmentationDataset(images_dir, masks_dir, transform, num_classes)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=True
    )
    return dataloader

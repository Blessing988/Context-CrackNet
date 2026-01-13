"""
Utility Functions for Context-CrackNet

This module provides utility functions for:
- Saving and loading model checkpoints
- Learning rate scheduling
"""

import torch
import os


def save_checkpoint(state, is_best, checkpoint_dir='checkpoints', filename='checkpoint.pth.tar'):
    """
    Save a model checkpoint.
    
    Args:
        state (dict): State dict containing model state, optimizer state, epoch, etc.
        is_best (bool): Whether this is the best model so far
        checkpoint_dir (str): Directory to save checkpoints
        filename (str): Checkpoint filename
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, best_path)


def load_checkpoint(model, optimizer=None, checkpoint_dir='checkpoints', filename='checkpoint.pth.tar'):
    """
    Load a model checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state into
        checkpoint_dir (str): Directory containing checkpoints
        filename (str): Checkpoint filename
        
    Returns:
        int: Epoch number from checkpoint, or 0 if not found
    """
    filepath = os.path.join(checkpoint_dir, filename)
    if os.path.isfile(filepath):
        print(f"Loading checkpoint '{filepath}'")
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint.get('epoch', 0)
        print(f"Loaded checkpoint '{filepath}' (epoch {epoch})")
        return epoch
    else:
        print(f"No checkpoint found at '{filepath}'")
        return 0


def adjust_learning_rate(optimizer, epoch, initial_lr, lr_decay_epoch=30):
    """
    Decay learning rate by factor of 10 every lr_decay_epoch epochs.
    
    Args:
        optimizer: PyTorch optimizer
        epoch (int): Current epoch
        initial_lr (float): Initial learning rate
        lr_decay_epoch (int): Epochs between learning rate decays
    """
    lr = initial_lr * (0.1 ** (epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

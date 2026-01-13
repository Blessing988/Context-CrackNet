"""
Loss Functions for Crack Segmentation

This module provides various loss functions for both binary and multiclass
crack segmentation:

Binary losses:
- BinaryDiceLoss: Standard Dice loss for binary segmentation
- LogCoshDiceLoss: Log-Cosh variant of Dice loss
- BinaryFocalLoss: Focal loss for handling class imbalance
- CombinedBinaryLoss: BCE + Dice combination

Multiclass losses:
- MulticlassDiceLoss: Dice loss for multiple classes
- MulticlassFocalLoss: Focal loss for multiclass
- CombinedMulticlassLoss: CE + Dice combination
- EnhancedMulticlassDiceLoss: Weighted Dice with class balancing
- ImprovedCombinedMulticlassLoss: With auxiliary loss support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryDiceLoss(nn.Module):
    """
    Binary Dice Loss for segmentation.
    
    Dice Loss = 1 - (2 * intersection + smooth) / (sum_pred + sum_target + smooth)
    
    Args:
        smooth (float): Smoothing factor to avoid division by zero
    """
    def __init__(self, smooth=1e-5):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = torch.sigmoid(logits)
        logits = logits.view(-1)
        targets = targets.view(-1)

        intersection = (logits * targets).sum()
        dice = (2. * intersection + self.smooth) / (logits.sum() + targets.sum() + self.smooth)
        loss = 1 - dice
        return loss


class LogCoshDiceLoss(nn.Module):
    """
    Log-Cosh Dice Loss for more stable training.
    
    Uses log(cosh(x)) which is smooth and acts like L1 for large errors.
    
    Args:
        smooth (float): Smoothing factor
    """
    def __init__(self, smooth=1e-5):
        super(LogCoshDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = torch.sigmoid(logits)
        logits = logits.view(-1)
        targets = targets.view(-1)

        intersection = (logits * targets).sum()
        dice = (2. * intersection + self.smooth) / (logits.sum() + targets.sum() + self.smooth)
        loss = torch.log(torch.cosh(1 - dice))
        return loss


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for handling class imbalance.
    
    Focal Loss = -alpha * (1 - pt)^gamma * log(pt)
    
    Args:
        alpha (float): Weighting factor for positive class
        gamma (float): Focusing parameter (higher = more focus on hard examples)
    """
    def __init__(self, alpha=0.25, gamma=2):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce_loss(logits, targets.float())
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class CombinedBinaryLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss.
    
    Combines BCE and Dice losses with configurable weights.
    
    Args:
        weight_bce (float): Weight for BCE loss
        weight_dice (float): Weight for Dice loss
        use_bce (bool): Whether to use BCE loss
        use_dice (bool): Whether to use Dice loss
    """
    def __init__(self, weight_bce=1.0, weight_dice=1.0, use_bce=True, use_dice=True):
        super(CombinedBinaryLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = BinaryDiceLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.use_dice = use_dice
        self.use_bce = use_bce

    def forward(self, logits, targets):
        if self.use_bce and not self.use_dice:
            loss = self.bce_loss(logits, targets.float())
        elif self.use_dice and not self.use_bce:
            loss = self.dice_loss(logits, targets)
        elif self.use_dice and self.use_bce:
            bce = self.bce_loss(logits, targets.float())
            dice = self.dice_loss(logits, targets)
            loss = self.weight_bce * bce + self.weight_dice * dice
        else:
            raise ValueError("At least one of use_bce or use_dice must be True")
        return loss


class MulticlassDiceLoss(nn.Module):
    """
    Multiclass Dice Loss.
    
    Computes Dice loss for each class and averages.
    
    Args:
        num_classes (int): Number of classes
        smooth (float): Smoothing factor
    """
    def __init__(self, num_classes, smooth=1e-5):
        super(MulticlassDiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, logits, targets):
        logits = torch.softmax(logits, dim=1)
        loss = 0
        for i in range(self.num_classes):
            logits_i = logits[:, i, :, :]
            targets_i = (targets == i).float()
            intersection = (logits_i * targets_i).sum()
            dice = (2. * intersection + self.smooth) / (logits_i.sum() + targets_i.sum() + self.smooth)
            loss += 1 - dice
        loss = loss / self.num_classes
        return loss


class MulticlassFocalLoss(nn.Module):
    """
    Multiclass Focal Loss for handling class imbalance.
    
    Args:
        alpha (float): Weighting factor
        gamma (float): Focusing parameter
    """
    def __init__(self, alpha=0.25, gamma=2):
        super(MulticlassFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce_loss(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CombinedMulticlassLoss(nn.Module):
    """
    Combined Cross Entropy and Multiclass Dice Loss.
    
    Args:
        num_classes (int): Number of classes
        weight_ce (float): Weight for cross entropy loss
        weight_dice (float): Weight for dice loss
        use_class_weights (bool): Whether to use class balancing weights
        use_bce (bool): Whether to use CE loss
        use_dice (bool): Whether to use Dice loss
    """
    def __init__(self, num_classes, weight_ce=0.5, weight_dice=0.5, 
                 use_class_weights=True, use_bce=True, use_dice=True):
        super(CombinedMulticlassLoss, self).__init__()
        
        # Default class weights (can be adjusted based on dataset)
        class_weights = {
            0: 0.31, 30: 2.17, 60: 2.94, 90: 3.48,
            120: 1.33, 150: 1.57, 180: 0.75
        }
        
        weights = torch.tensor(
            [class_weights[i] for i in sorted(class_weights.keys())], 
            dtype=torch.float
        )

        if use_class_weights:
            self.ce_loss = nn.CrossEntropyLoss(weight=weights.cuda() if torch.cuda.is_available() else weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        
        self.dice_loss = MulticlassDiceLoss(num_classes)
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.use_dice = use_dice
        self.use_bce = use_bce

    def forward(self, logits, targets):
        if self.use_bce and not self.use_dice:
            loss = self.ce_loss(logits, targets)
        elif self.use_dice and not self.use_bce:
            loss = self.dice_loss(logits, targets)
        elif self.use_dice and self.use_bce:
            ce = self.ce_loss(logits, targets)
            dice = self.dice_loss(logits, targets)
            loss = self.weight_ce * ce + self.weight_dice * dice
        else:
            raise ValueError("At least one of use_bce or use_dice must be True")
        return loss


class EnhancedMulticlassDiceLoss(nn.Module):
    """
    Enhanced Multiclass Dice Loss with automatic class weighting.
    
    Weights classes based on inverse frequency for better handling
    of imbalanced datasets.
    
    Args:
        num_classes (int): Number of classes
        smooth (float): Smoothing factor
    """
    def __init__(self, num_classes, smooth=1e-5):
        super(EnhancedMulticlassDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        
    def forward(self, logits, targets):
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Apply softmax to logits
        probs = F.softmax(logits, dim=1)
        
        # Calculate Dice score for each class
        dice_scores = []
        for class_idx in range(self.num_classes):
            class_probs = probs[:, class_idx]
            class_targets = targets_one_hot[:, class_idx]
            
            intersection = (class_probs * class_targets).sum(dim=(1, 2))
            union = class_probs.sum(dim=(1, 2)) + class_targets.sum(dim=(1, 2))
            
            # Apply class-specific weighting based on pixel frequency
            class_weight = 1.0 / (class_targets.sum(dim=(1, 2)).mean() + self.smooth)
            
            dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
            weighted_dice = (1.0 - dice_score.mean()) * class_weight
            dice_scores.append(weighted_dice)
            
        return torch.stack(dice_scores).mean()


class ImprovedCombinedMulticlassLoss(nn.Module):
    """
    Improved Combined Loss with auxiliary output support.
    
    Supports models with auxiliary outputs for deep supervision.
    
    Args:
        num_classes (int): Number of classes
        weight_ce (float): Weight for cross entropy loss
        weight_dice (float): Weight for dice loss
        weight_aux (float): Weight for auxiliary losses
    """
    def __init__(self, num_classes, weight_ce=0.5, weight_dice=0.5, weight_aux=0.3):
        super(ImprovedCombinedMulticlassLoss, self).__init__()
        
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.dice_loss = EnhancedMulticlassDiceLoss(num_classes)
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_aux = weight_aux
        
    def forward(self, outputs, targets):
        if isinstance(outputs, tuple):
            main_out, aux1, aux2 = outputs
            
            # Main loss
            ce_loss = self.ce_loss(main_out, targets)
            dice_loss = self.dice_loss(main_out, targets)
            main_loss = self.weight_ce * ce_loss.mean() + self.weight_dice * dice_loss
            
            # Auxiliary losses
            aux1_loss = self.ce_loss(aux1, targets).mean()
            aux2_loss = self.ce_loss(aux2, targets).mean()
            
            # Combine losses with auxiliary weight
            total_loss = main_loss + self.weight_aux * (aux1_loss + aux2_loss)
            return total_loss
        else:
            ce_loss = self.ce_loss(outputs, targets)
            dice_loss = self.dice_loss(outputs, targets)
            return self.weight_ce * ce_loss.mean() + self.weight_dice * dice_loss

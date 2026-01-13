import torch.nn as nn
import torch
import torch.nn.functional as F
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class BinaryDiceLoss(nn.Module):
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
    def __init__(self, weight_bce=1.0, weight_dice=1.0):
        super(CombinedBinaryLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = BinaryDiceLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

        self.use_dice = config['model']['use_dice']
        self.use_bce = config['model']['use_bce']

    def forward(self, logits, targets):
        if self.use_bce and not self.use_dice:
            loss = self.bce_loss(logits, targets.float())
        if self.use_dice and not self.use_bce:
            loss = self.dice_loss(logits, targets)

        if self.use_dice and self.use_bce:
            bce = self.bce_loss(logits, targets.float())
            dice = self.dice_loss(logits, targets)
            loss = self.weight_bce * bce + self.weight_dice * dice

        return loss


class MulticlassDiceLoss(nn.Module):
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
    def __init__(self, num_classes, weight_ce=0.5, weight_dice=0.5, use_class_weights=True):
        super(CombinedMulticlassLoss, self).__init__()
        
        class_weights = {
            0: 0.31,
            30: 2.17,
            60: 2.94,
            90: 3.48,
            120: 1.33,
            150: 1.57,
            180: 0.75}
        
        weights = torch.tensor([class_weights[i] for i in sorted(class_weights.keys())], dtype=torch.float)

        if use_class_weights:
            self.ce_loss = nn.CrossEntropyLoss(weight = weights.to('cuda'))
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        
        self.dice_loss = MulticlassDiceLoss(num_classes)
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        
        self.use_dice = config['model']['use_dice']
        self.use_bce = config['model']['use_bce']

    def forward(self, logits, targets):
        # print('Cross Entropy Loss', ce)
        # print('Dice Loss', dice)
        
        if self.use_bce and not self.use_dice:
            loss = self.ce_loss(logits, targets)
        if self.use_dice and not self.use_bce:
            loss = self.dice_loss(logits, targets)

        if self.use_dice and self.use_bce:
            ce = self.ce_loss(logits, targets)
            dice = self.dice_loss(logits, targets)
            loss = self.weight_ce * ce + self.weight_dice * dice
        
        return loss


class EnhancedMulticlassDiceLoss(nn.Module):
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
    def __init__(self, num_classes, weight_ce=0.5, weight_dice=0.5, weight_aux=0.3):
        super(ImprovedCombinedMulticlassLoss, self).__init__()
        
        # Calculate class weights based on inverse frequency
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

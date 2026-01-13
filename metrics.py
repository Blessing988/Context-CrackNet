import torch
import numpy as np
import os
import csv


def calculate_iou(pred_mask, true_mask, num_classes):
    ious = []
    pred_mask = pred_mask.view(-1)
    true_mask = true_mask.view(-1)
    if num_classes == 1:
        pred_inds = (pred_mask == 1)
        true_inds = (true_mask == 1)
        intersection = (pred_inds & true_inds).sum().float()
        union = (pred_inds | true_inds).sum().float()
        if union == 0:
            ious.append(float('nan'))  # Ignore if there is no ground truth
        else:
            ious.append((intersection / union).item())
            
    else:
        for cls in range(num_classes):
            pred_inds = (pred_mask == cls)
            true_inds = (true_mask == cls)
            intersection = (pred_inds & true_inds).sum().float()
            union = (pred_inds | true_inds).sum().float()
            if union == 0:
                ious.append(float('nan'))  # Ignore if there is no ground truth
            else:
                ious.append((intersection / union).item())
                
    return ious


def calculate_dice(pred_mask, true_mask, num_classes):
    dices = []
    pred_mask = pred_mask.view(-1)
    true_mask = true_mask.view(-1)
    if num_classes == 1:
        pred_inds = (pred_mask == 1)
        true_inds = (true_mask == 1)
        intersection = (pred_inds & true_inds).sum().float()
        total = pred_inds.sum() + true_inds.sum()
        if total == 0:
            dices.append(float('nan'))
        else:
            dices.append((2 * intersection / total).item())
    
    else:
        for cls in range(num_classes):
            pred_inds = (pred_mask == cls)
            true_inds = (true_mask == cls)
            intersection = (pred_inds & true_inds).sum().float()
            total = pred_inds.sum() + true_inds.sum()
            if total == 0:
                dices.append(float('nan'))
            else:
                dices.append((2 * intersection / total).item())
    return dices


def calculate_precision_recall(pred_mask, true_mask, num_classes):
    precisions = []
    recalls = []
    pred_mask = pred_mask.view(-1)
    true_mask = true_mask.view(-1)
    
    if num_classes == 1 :
        pred_inds = (pred_mask == 1)
        true_inds = (true_mask == 1)
        true_positive = (pred_inds & true_inds).sum().float()
        predicted_positive = pred_inds.sum().float()
        actual_positive = true_inds.sum().float()
        if predicted_positive == 0:
            precisions.append(float('nan'))
        else:
            precisions.append((true_positive / predicted_positive).item())
        if actual_positive == 0:
            recalls.append(float('nan'))
        else:
            recalls.append((true_positive / actual_positive).item())
            
    else:
        for cls in range(num_classes):
            pred_inds = (pred_mask == cls)
            true_inds = (true_mask == cls)
            true_positive = (pred_inds & true_inds).sum().float()
            predicted_positive = pred_inds.sum().float()
            actual_positive = true_inds.sum().float()
            if predicted_positive == 0:
                precisions.append(float('nan'))
            else:
                precisions.append((true_positive / predicted_positive).item())
            if actual_positive == 0:
                recalls.append(float('nan'))
            else:
                recalls.append((true_positive / actual_positive).item())
                
    return precisions, recalls


def calculate_f1_score(precisions, recalls):
    f1_scores = []
    for p, r in zip(precisions, recalls):
        if p == 0 or r == 0 or p != p or r != r:  # Check for zero or NaN
            f1_scores.append(float('nan'))
        else:
            f1_scores.append(2 * p * r / (p + r))
    return f1_scores


def evaluate_metrics(pred_mask, true_mask, num_classes):
    ious = calculate_iou(pred_mask, true_mask, num_classes)
    dices = calculate_dice(pred_mask, true_mask, num_classes)
    precisions, recalls = calculate_precision_recall(pred_mask, true_mask, num_classes)
    f1_scores = calculate_f1_score(precisions, recalls)

    metrics = {
        'IoU': np.nanmean(ious) if len(ious) > 0 else np.nan,
        'Dice': np.nanmean(dices) if len(dices) > 0 else np.nan,
        'Precision': np.nanmean(precisions) if len(precisions) > 0 else np.nan,
        'Recall': np.nanmean(recalls) if len(recalls) > 0 else np.nan,
        'F1 Score': np.nanmean(f1_scores) if len(f1_scores) > 0 else np.nan
    }

    return metrics


def save_metrics(epoch, train_loss, train_metrics, val_loss, val_metrics, output_csv):
    # Create the directory if it doesn't exist
    output_dir = os.path.dirname(output_csv)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert tensors to scalar values using .item()
    train_metrics = [m.item() if isinstance(m, torch.Tensor) else m for m in train_metrics]
    val_metrics = [m.item() if isinstance(m, torch.Tensor) else m for m in val_metrics]

    # Check if the CSV file exists, if not, write the header
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            # Write the header row
            header = [
                'Epoch',
                'Train Loss', 'Train IoU', 'Train Dice Score', 'Train Precision', 'Train Recall', 'Train F1 score',
                'Val Loss', 'Val IoU', 'Val Dice Score', 'Val Precision', 'Val Recall', 'Val F1 score'
            ]
            writer.writerow(header)
        # Write the metrics row
        writer.writerow([epoch] + [train_loss] +  train_metrics + [val_loss] + val_metrics)


## New updated code:

import torch
import numpy as np
from collections import defaultdict

class SegmentationMetrics:
    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.reset()
        
    def reset(self):
        self.metrics_sum = defaultdict(lambda: np.zeros(self.num_classes))
        self.metrics_count = defaultdict(lambda: np.zeros(self.num_classes))
        
    def update(self, pred_mask, true_mask):
        """Update metrics for current batch."""
        pred_mask = pred_mask.view(-1)
        true_mask = true_mask.view(-1)
        
        # Calculate metrics for each class
        for cls in range(self.num_classes):
            pred_inds = (pred_mask == cls)
            true_inds = (true_mask == cls)
            
            # Basic counts
            true_positive = (pred_inds & true_inds).sum().float()
            false_positive = (pred_inds & ~true_inds).sum().float()
            false_negative = (~pred_inds & true_inds).sum().float()
            
            # Update confusion matrix elements
            self.metrics_sum['tp'][cls] += true_positive.item()
            self.metrics_sum['fp'][cls] += false_positive.item()
            self.metrics_sum['fn'][cls] += false_negative.item()
            
            # Count valid samples for this class
            if true_inds.sum() > 0:
                self.metrics_count['samples'][cls] += 1
    
    def compute_class_weights(self):
        """Compute class weights based on frequency."""
        total_pixels = sum(self.metrics_sum['tp'] + self.metrics_sum['fn'])
        class_frequencies = (self.metrics_sum['tp'] + self.metrics_sum['fn']) / total_pixels
        return 1 / (class_frequencies + 1e-5)
    
    def compute_metrics(self):
        """Compute final metrics with class-specific and weighted averages."""
        metrics = {}
        class_weights = self.compute_class_weights()
        
        # Per-class metrics
        for cls in range(self.num_classes):
            tp = self.metrics_sum['tp'][cls]
            fp = self.metrics_sum['fp'][cls]
            fn = self.metrics_sum['fn'][cls]
            
            # Handle division by zero
            union = tp + fp + fn
            total = tp + fp + fn
            
            if union > 0:
                iou = tp / union
            else:
                iou = 0.0
                
            if total > 0:
                dice = 2 * tp / (2 * tp + fp + fn)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            else:
                dice = precision = recall = f1 = 0.0
            
            metrics[f'{self.class_names[cls]}_iou'] = iou
            metrics[f'{self.class_names[cls]}_dice'] = dice
            metrics[f'{self.class_names[cls]}_precision'] = precision
            metrics[f'{self.class_names[cls]}_recall'] = recall
            metrics[f'{self.class_names[cls]}_f1'] = f1
        
        # Compute weighted averages
        weighted_metrics = {
            'weighted_iou': 0.0,
            'weighted_dice': 0.0,
            'weighted_f1': 0.0
        }
        
        for cls in range(self.num_classes):
            weight = class_weights[cls]
            weighted_metrics['weighted_iou'] += weight * metrics[f'{self.class_names[cls]}_iou']
            weighted_metrics['weighted_dice'] += weight * metrics[f'{self.class_names[cls]}_dice']
            weighted_metrics['weighted_f1'] += weight * metrics[f'{self.class_names[cls]}_f1']
        
        # Normalize weighted metrics
        total_weight = sum(class_weights)
        for key in weighted_metrics:
            weighted_metrics[key] /= total_weight
            metrics[key] = weighted_metrics[key]
        
        return metrics

def evaluate_batch(pred_mask, true_mask, num_classes):
    """Single batch evaluation for use during training."""
    metrics = SegmentationMetrics(num_classes)
    metrics.update(pred_mask, true_mask)
    return metrics.compute_metrics()

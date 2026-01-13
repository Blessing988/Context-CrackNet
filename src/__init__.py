# Context-CrackNet: Crack Segmentation with Context-Aware Global Mechanisms
"""
Context-CrackNet is a novel deep learning framework for crack segmentation
that combines a ResNet50 encoder with Linformer-based attention and
attention-gated skip connections.
"""

from src.models import Context_CrackNet, Context_CrackNet_ablation, create_model
from src.data import get_transforms, get_dataloader, SegmentationDataset
from src.losses import CombinedBinaryLoss, CombinedMulticlassLoss
from src.metrics import evaluate_metrics, save_metrics

__version__ = "1.0.0"
__author__ = "Blessing Agyei Kyem"

__all__ = [
    "Context_CrackNet",
    "Context_CrackNet_ablation", 
    "create_model",
    "get_transforms",
    "get_dataloader",
    "SegmentationDataset",
    "CombinedBinaryLoss",
    "CombinedMulticlassLoss",
    "evaluate_metrics",
    "save_metrics",
]

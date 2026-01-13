# Losses package for Context-CrackNet
"""
This package contains loss functions for crack segmentation training.
"""

from src.losses.losses import (
    BinaryDiceLoss,
    LogCoshDiceLoss,
    BinaryFocalLoss,
    CombinedBinaryLoss,
    MulticlassDiceLoss,
    MulticlassFocalLoss,
    CombinedMulticlassLoss,
    EnhancedMulticlassDiceLoss,
    ImprovedCombinedMulticlassLoss,
)

__all__ = [
    "BinaryDiceLoss",
    "LogCoshDiceLoss",
    "BinaryFocalLoss",
    "CombinedBinaryLoss",
    "MulticlassDiceLoss",
    "MulticlassFocalLoss",
    "CombinedMulticlassLoss",
    "EnhancedMulticlassDiceLoss",
    "ImprovedCombinedMulticlassLoss",
]

"""
TTV (Train-Test-Verify) Package for Gait Recognition.
Exposes the core model, data, and configuration classes for easy import.
"""

from .config import CFG, Config
from .models import (
    ModelDims,
    FeatureExtractor,
    SixStreamFusionNet,
    BatchHardTripletLoss,
)
from .data import (
    create_dataloaders,
    GaitDataset,
    BalancedBatchSampler,
)

__all__ = [
    "CFG",
    "Config",
    "ModelDims",
    "FeatureExtractor",
    "SixStreamFusionNet",
    "BatchHardTripletLoss",
    "create_dataloaders",
    "GaitDataset",
    "BalancedBatchSampler",
]

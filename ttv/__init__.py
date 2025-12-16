"""
TTV (Train-Test-Verify) Package for Gait Recognition.
Exposes the core model, data, and configuration classes for easy import.
"""

from .config import CFG, Config
from .models import (
    HybridGaitTransformer,
    BatchHardTripletLoss,
)
from .data import (
    create_dataloaders,
    WindowDataset,
    BalancedBatchSampler,
)

__all__ = [
    "CFG",
    "Config",
    "HybridGaitTransformer",
    "BatchHardTripletLoss",
    "create_dataloaders",
    "WindowDataset",
    "BalancedBatchSampler",
]

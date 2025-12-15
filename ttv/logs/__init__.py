"""
TTV (Train-Test-Verify) Package for Gait Recognition.
Exposes the core model, data, and configuration classes for easy import.
"""

from .config import CFG, Config, make_cfg
from .models import SiameseFusion, SixStreamFusionNet, TripletLoss, FeatureExtractor
from .data import SixStreamGaitDataset, create_dataloaders
from .tgm import main as train_model

__all__ = [
	"CFG",
	"Config",
	"make_cfg",
	"SiameseFusion",
	"SixStreamFusionNet",
	"TripletLoss",
	"FeatureExtractor",
	"SixStreamGaitDataset",
	"create_dataloaders",
	"train_model",
]
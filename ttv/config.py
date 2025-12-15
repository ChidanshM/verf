"""Configuration + run signatures for gait model training.

Keep *all* tunable values in one place so you don't have to edit multiple modules.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class Config:
	# Repro
	seed: int = 42 # THIS IS THE NUMBER OF LIFE

	# Training
	batch_size: int = 64
	epochs: int = 30
	lr: float = 1e-4
	weight_decay: float = 1e-2  # aligns with AdamW default used in your script

	# Loss
	margin: float = 0.1

	# Data / windowing
	input_channels: int = 6
	window_size: int = 200
	stride: int = 50

	# Sensor streams (order matters for deterministic concatenation)
	streams: Tuple[str, ...] = (
		"Pelvis",
		"Upper_Spine",
		"Shank_LT",
		"Foot_LT",
		"Shank_RT",
		"Foot_RT",
	)

	# Model dims
	feat_dim: int = 64      # per-stream feature dim produced by FeatureExtractor
	hidden_dim: int = 128   # fusion MLP hidden dim
	emb_dim: int = 64       # final embedding dim

	# Scheduler defaults (ReduceLROnPlateau)
	sched_mode: str = "min"
	sched_factor: float = 0.5
	sched_patience: int = 2


# Default config used across modules
CFG = Config()


def make_cfg(**overrides: Any) -> Config:
	"""Create a modified Config without mutating the global CFG.

	Example:
		cfg = make_cfg(window_size=256, stride=64)
	"""
	return replace(CFG, **overrides)


def build_model_signature(cfg: Config) -> Dict[str, Any]:
	"""Signature used to ensure checkpoint architecture compatibility."""
	fusion_in = cfg.feat_dim * len(cfg.streams)
	return {
		"model_name": "SiameseFusion/SixStreamFusionNet",
		"input_channels": cfg.input_channels,
		"window_size": cfg.window_size,
		"streams": list(cfg.streams),
		"feat_dim": cfg.feat_dim,
		"hidden_dim": cfg.hidden_dim,
		"embedding_dim": cfg.emb_dim,
		"fusion_in": fusion_in,
	}


def build_training_signature(
	optimizer_type: str,
	scheduler_type: str,
) -> Dict[str, Any]:
	"""Signature used to ensure checkpoint training-state compatibility.

	Per your requirement, this only checks optimiser *type* and scheduler *type*.
	"""
	return {
		"optimizer_type": optimizer_type,
		"scheduler_type": scheduler_type,
	}

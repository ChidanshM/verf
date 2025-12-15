# verf/ttv/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn

from .config import Config


@dataclass(frozen=True)
class ModelDims:
	"""Optional override container for model dims."""
	feat_dim: int
	hidden_dim: int
	emb_dim: int
	dropout: float = 0.5


class FeatureExtractor(nn.Module):
	"""
	Single-stream 1D CNN.
	Input:  (B, C, T)
	Output: (B, feat_dim)
	"""

	def __init__(self, input_channels: int, feat_dim: int, mid_channels: int = 16):
		super().__init__()
		self.net = nn.Sequential(
			# Layer 1: 16 Filters
			nn.Conv1d(input_channels, mid_channels, kernel_size=7, padding=3),
			nn.InstanceNorm1d(feat_dim, affine=True),
			nn.ReLU(),
			nn.MaxPool1d(2),

			# Layer 2: 32 Filters (2 * mid_channels)

			nn.Conv1d(mid_channels, feat_dim, kernel_size=7, padding=3),
			nn.InstanceNorm1d(feat_dim, affine=True),
			nn.ReLU(),
			nn.AdaptiveAvgPool1d(1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x).squeeze(-1)

# In verf/ttv/models.py

class SixStreamFusionNet(nn.Module):
	"""
	SHARED WEIGHTS VERSION
	Forces the model to learn generic motion features (Gait) 
	instead of specific sensor artifacts.
	"""
	def __init__(self, cfg: Config, dims: ModelDims | None = None):
		super().__init__()
		self.cfg = cfg
		# We still expect these 6 inputs
		self.streams = ['Pelvis', 'Upper_Spine', 'Shank_LT', 'Foot_LT', 'Shank_RT', 'Foot_RT']

		if dims is None:
			# Fallback to defaults if not provided
			dims = ModelDims(
				feat_dim=cfg.feat_dim,
				hidden_dim=cfg.hidden_dim,
				emb_dim=cfg.emb_dim,
				dropout=0.5
			)
		self.dims = dims

		# --- THE KEY CHANGE ---
		# Instead of 6 separate networks, we initialize ONE network.
		# This reduces parameters by 6x and forces generalization.
		self.shared_feature_extractor = FeatureExtractor(
			input_channels=cfg.input_channels, 
			feat_dim=self.dims.feat_dim
		)

		# Fusion layer calculates how to combine the 6 views
		fusion_input_size = self.dims.feat_dim * len(self.streams)

		self.fusion = nn.Sequential(
			nn.Linear(fusion_input_size, self.dims.hidden_dim),
			nn.ReLU(),
			nn.Dropout(self.dims.dropout),
			nn.Linear(self.dims.hidden_dim, self.dims.emb_dim),
		)

	def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
		# 1. Process every sensor through the SAME network
		feats = []
		for sensor_name in self.streams:
			# Reuse 'shared_feature_extractor' for everyone
			x = inputs[sensor_name]
			out = self.shared_feature_extractor(x)
			feats.append(out)
		
		# 2. Concatenate results (Order matters! Keep consistent)
		combined = torch.cat(feats, dim=1)
		
		# 3. Fuse them into a final ID vector
		embedding = self.fusion(combined)
		
		# 4. Normalize (Critical for stability)
		return torch.nn.functional.normalize(embedding, p=2, dim=1)

# class SixStreamFusionNet(nn.Module):
# 	"""
# 	Multi-stream fusion network.
# 	Expects: inputs = {stream_name: Tensor[B, C, T]}
# 	Returns: embedding Tensor[B, emb_dim]
# 	"""

# 	def __init__(self, cfg: Config, dims: ModelDims | None = None):
# 		super().__init__()
# 		self.cfg = cfg
# 		self.streams: List[str] = list(cfg.streams)

# 		# Use cfg dims by default; allow override via ModelDims if needed
# 		if dims is None:
# 			dims = ModelDims(
# 				feat_dim=cfg.feat_dim,
# 				hidden_dim=cfg.hidden_dim,
# 				emb_dim=cfg.emb_dim,
# 				dropout=0.5,
# 			)
# 		self.dims = dims

# 		self.branches = nn.ModuleDict({
# 			s: FeatureExtractor(
# 				input_channels=cfg.input_channels,
# 				feat_dim=self.dims.feat_dim,
# 			)
# 			for s in self.streams
# 		})

# 		fusion_in = self.dims.feat_dim * len(self.streams)

# 		self.fusion = nn.Sequential(
# 			nn.Linear(fusion_in, self.dims.hidden_dim),
# 			nn.ReLU(),
# 			nn.Dropout(self.dims.dropout),
# 			nn.Linear(self.dims.hidden_dim, self.dims.emb_dim),
# 		)

# 	def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
# 		# Deterministic order: always use cfg.streams, not dict order
# 		feats = [self.branches[s](inputs[s]) for s in self.streams]
# 		combined = torch.cat(feats, dim=1)
# 		return self.fusion(combined)


class SiameseFusion(nn.Module):
	"""Wrapper that returns embeddings for (anchor, positive, negative)."""

	def __init__(self, cfg: Config, dims: ModelDims | None = None):
		super().__init__()
		self.backbone = SixStreamFusionNet(cfg=cfg, dims=dims)

	def forward(
		self,
		a: Dict[str, torch.Tensor],
		p: Dict[str, torch.Tensor],
		n: Dict[str, torch.Tensor],
	):
		ea, ep, en = self.backbone(a), self.backbone(p), self.backbone(n)
		# This restricts vectors to length 1.0, stabilizing training
		ea = torch.nn.functional.normalize(ea, p=2, dim=1)
		ep = torch.nn.functional.normalize(ep, p=2, dim=1)
		en = torch.nn.functional.normalize(en, p=2, dim=1)
		
		# ----------------------------------
		return ea, ep, en


class TripletLoss(nn.Module):
	def __init__(self, margin: float = 0.2):
		super().__init__()
		self.margin = float(margin)

	def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
		dist_pos = torch.pow(anchor - positive, 2).sum(dim=1)
		dist_neg = torch.pow(anchor - negative, 2).sum(dim=1)
		return torch.relu(dist_pos - dist_neg + self.margin).mean()


__all__ = [
	"ModelDims",
	"FeatureExtractor",
	"SixStreamFusionNet",
	"SiameseFusion",
	"TripletLoss",
]

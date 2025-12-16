# verf/ttv/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F  # Explicit import for cleaner code

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
			nn.InstanceNorm1d(mid_channels, affine=True), # Corrected feat_dim -> mid_channels here for safety
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


class SixStreamFusionNet(nn.Module):
	"""
	SHARED WEIGHTS VERSION
	Forces the model to learn generic motion features (Gait) 
	instead of specific sensor artifacts.
	"""
	def __init__(self, cfg: Config, dims: ModelDims | None = None):
		super().__init__()
		self.cfg = cfg
		
		# FIX: Use the config streams, do not hardcode!
		self.streams = list(cfg.streams)

		if dims is None:
			dims = ModelDims(
				feat_dim=cfg.feat_dim,
				hidden_dim=cfg.hidden_dim,
				emb_dim=cfg.emb_dim,
				dropout=0.5
			)
		self.dims = dims

		# Shared Feature Extractor
		self.shared_feature_extractor = FeatureExtractor(
			input_channels=cfg.input_channels, 
			feat_dim=self.dims.feat_dim
		)

		# Fusion layer inputs = feat_dim * number of streams in config
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
		
		# Iterate over the streams defined in Config
		for sensor_name in self.streams:
			if sensor_name not in inputs:
				raise KeyError(f"Expected stream '{sensor_name}' from Config, but it is missing in inputs.")
			
			x = inputs[sensor_name]
			out = self.shared_feature_extractor(x)
			feats.append(out)
		
		# 2. Concatenate results (Order matches cfg.streams)
		combined = torch.cat(feats, dim=1)
		
		# 3. Fuse into final embedding
		embedding = self.fusion(combined)
		
		# 4. Normalize (Critical for stability)
		return F.normalize(embedding, p=2, dim=1)


# class SiameseFusion(nn.Module):
#     """Wrapper that returns embeddings for (anchor, positive, negative)."""

#     def __init__(self, cfg: Config, dims: ModelDims | None = None):
#         super().__init__()
#         self.backbone = SixStreamFusionNet(cfg=cfg, dims=dims)

#     def forward(
#         self,
#         a: Dict[str, torch.Tensor],
#         p: Dict[str, torch.Tensor],
#         n: Dict[str, torch.Tensor],
#     ):
#         # Backbone already normalizes output, so we don't need to do it again here
#         ea = self.backbone(a)
#         ep = self.backbone(p)
#         en = self.backbone(n)
		
#         return ea, ep, en


class TripletLoss(nn.Module):
	def __init__(self, margin: float = 0.2):
		super().__init__()
		self.margin = float(margin)

	def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
		# Distance calculation
		dist_pos = torch.pow(anchor - positive, 2).sum(dim=1)
		dist_neg = torch.pow(anchor - negative, 2).sum(dim=1)
		
		# Loss
		return torch.relu(dist_pos - dist_neg + self.margin).mean()

# verf/ttv/models.py (Add this new class)

class BatchHardTripletLoss(nn.Module):
	def __init__(self, margin: float = 0.5):
		super().__init__()
		self.margin = margin

	def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		"""
		embeddings: (B, dim)
		labels: (B)
		"""
		# 1. Compute Pairwise Distance Matrix (B x B)
		# |x-y|^2 = |x|^2 + |y|^2 - 2xy
		dot_product = torch.matmul(embeddings, embeddings.t())
		square_norm = torch.diag(dot_product)
		# Distances squared: (B, 1) + (1, B) - 2(B, B)
		dist_sq = square_norm.unsqueeze(1) + square_norm.unsqueeze(0) - 2.0 * dot_product
		
		# Handle numerical stability (ReLU prevents negative zeros, Sqrt is safe)
		distance_matrix = torch.sqrt(torch.relu(dist_sq) + 1e-16)

		# 2. Identify Positives and Negatives Masks
		# labels_eq: (B, B) matrix where True if labels[i] == labels[j]
		labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
		mask_pos = labels_equal.float()
		mask_neg = (~labels_equal).float()

		# 3. Get Hardest Positive (Max distance among same class)
		# We multiply by mask so negatives become 0. 
		# But we want max, so 0 is fine (distances are >= 0).
		hardest_positive_dist = (distance_matrix * mask_pos).max(dim=1)[0]

		# 4. Get Hardest Negative (Min distance among diff class)
		# We add a huge value to positives so they aren't picked as min
		max_dist = distance_matrix.max().detach()
		hardest_negative_dist = (distance_matrix + max_dist * mask_pos).min(dim=1)[0]

		# 5. Compute Loss
		# ReLU( Hard_Pos - Hard_Neg + Margin )
		triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
		
		return triplet_loss.mean()


__all__ = [
	"ModelDims",
	"FeatureExtractor",
	"SixStreamFusionNet",
	"SiameseFusion",
	"TripletLoss",
	"BatchHardTripletLoss",
]


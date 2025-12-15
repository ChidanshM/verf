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
			nn.Conv1d(input_channels, mid_channels, kernel_size=3, padding=1),
			nn.BatchNorm1d(mid_channels),
			nn.ReLU(),
			nn.MaxPool1d(2),

			# Layer 2: 32 Filters (2 * mid_channels)

			nn.Conv1d(mid_channels, feat_dim, kernel_size=3, padding=1),
			nn.BatchNorm1d(feat_dim),
			nn.ReLU(),
			nn.AdaptiveAvgPool1d(1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x).squeeze(-1)


class SixStreamFusionNet(nn.Module):
	"""
	Multi-stream fusion network.
	Expects: inputs = {stream_name: Tensor[B, C, T]}
	Returns: embedding Tensor[B, emb_dim]
	"""

	def __init__(self, cfg: Config, dims: ModelDims | None = None):
		super().__init__()
		self.cfg = cfg
		self.streams: List[str] = list(cfg.streams)

		# Use cfg dims by default; allow override via ModelDims if needed
		if dims is None:
			dims = ModelDims(
				feat_dim=cfg.feat_dim,
				hidden_dim=cfg.hidden_dim,
				emb_dim=cfg.emb_dim,
				dropout=0.5,
			)
		self.dims = dims

		self.branches = nn.ModuleDict({
			s: FeatureExtractor(
				input_channels=cfg.input_channels,
				feat_dim=self.dims.feat_dim,
			)
			for s in self.streams
		})

		fusion_in = self.dims.feat_dim * len(self.streams)

		self.fusion = nn.Sequential(
			nn.Linear(fusion_in, self.dims.hidden_dim),
			nn.ReLU(),
			nn.Dropout(self.dims.dropout),
			nn.Linear(self.dims.hidden_dim, self.dims.emb_dim),
		)

	def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
		# Deterministic order: always use cfg.streams, not dict order
		feats = [self.branches[s](inputs[s]) for s in self.streams]
		combined = torch.cat(feats, dim=1)
		return self.fusion(combined)


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

# verf/ttv/models.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .config import Config

class HybridGaitTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.streams = list(cfg.streams)
        
        # Calculate total input channels (e.g., 6 sensors * 9 axes = 54)
        total_input_channels = len(self.streams) * cfg.input_channels
        
        # --- Block A: Spatial Feature Extractor (CNN Stem) ---
        self.cnn_stem = nn.Sequential(
            # Layer 1: Reduce noise, halve time dimension
            nn.Conv1d(total_input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 2: Refine features
            nn.Conv1d(64, cfg.feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(cfg.feat_dim),
            nn.ReLU()
        )
        
        # --- Block B: Temporal Attention (Transformer) ---
        # The CNN stride=2 reduced time from 500 -> 250
        reduced_seq_len = cfg.window_size // 2
        
        # Positional Encoding (Learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, cfg.feat_dim, reduced_seq_len))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.feat_dim,
            nhead=cfg.n_head,
            dim_feedforward=512,
            dropout=0.2,
            activation='gelu',
            batch_first=True,
            norm_first=True # Better stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        
        # --- Block C: Authentication Head ---
        self.fc_head = nn.Linear(cfg.feat_dim, cfg.emb_dim)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 1. Fuse Dictionary into Tensor
        # Inputs are (B, C_sensor, T). Stack channel-wise -> (B, C_total, T)
        # Order matters! Must match cfg.streams
        sensor_list = [inputs[s] for s in self.streams]
        x = torch.cat(sensor_list, dim=1) 
        
        # 2. CNN Spatial Extraction
        # Input: (Batch, 54, 500) -> Output: (Batch, 128, 250)
        x = self.cnn_stem(x)
        
        # 3. Add Positional Encoding
        x = x + self.pos_embedding
        
        # 4. Transformer expects (Batch, Time, Channels)
        x = x.permute(0, 2, 1) # (B, 128, 250) -> (B, 250, 128)
        
        # 5. Temporal Attention
        x = self.transformer(x)
        
        # 6. Global Average Pooling (Collapse Time)
        # (B, 250, 128) -> (B, 128)
        x = x.mean(dim=1)
        
        # 7. Project to Embedding
        x = self.fc_head(x)
        
        # 8. L2 Normalize (Hypersphere projection for Triplet Loss)
        return F.normalize(x, p=2, dim=1)

# Keep the Loss Function
class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        dist_sq = square_norm.unsqueeze(1) + square_norm.unsqueeze(0) - 2.0 * dot_product
        distance_matrix = torch.sqrt(torch.relu(dist_sq) + 1e-16)

        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_pos = labels_equal.float()
        
        # Hardest Positive: Max distance among same class
        hardest_positive_dist = (distance_matrix * mask_pos).max(dim=1)[0]

        # Hardest Negative: Min distance among diff class
        max_dist = distance_matrix.max().detach()
        hardest_negative_dist = (distance_matrix + max_dist * mask_pos).min(dim=1)[0]

        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        return triplet_loss.mean()

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
        
        total_input_channels = len(self.streams) * cfg.input_channels
        
        # --- STRATEGY 3: Higher Dropout in CNN ---
        self.cnn_stem = nn.Sequential(
            nn.Conv1d(total_input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4), # Was 0.3 -> Now 0.4
            
            nn.Conv1d(64, cfg.feat_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(cfg.feat_dim),
            nn.ReLU()
        )
        
        reduced_seq_len = cfg.window_size // 4 
        
        self.pos_embedding = nn.Parameter(torch.randn(1, cfg.feat_dim, reduced_seq_len))
        
        # --- STRATEGY 3: Higher Dropout in Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.feat_dim,
            nhead=cfg.n_head,
            dim_feedforward=512,
            dropout=0.3, # Was 0.2 -> Now 0.3
            activation='gelu',
            batch_first=True,
            norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        
        self.fc_head = nn.Linear(cfg.feat_dim, cfg.emb_dim)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        sensor_list = [inputs[s] for s in self.streams]
        x = torch.cat(sensor_list, dim=1) 
        
        x = self.cnn_stem(x)
        
        if x.shape[2] != self.pos_embedding.shape[2]:
             x = x + self.pos_embedding[:, :, :x.shape[2]]
        else:
             x = x + self.pos_embedding
        
        x = x.permute(0, 2, 1) 
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc_head(x)
        
        return F.normalize(x, p=2, dim=1)

# Keep Loss Same
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
        
        hardest_positive_dist = (distance_matrix * mask_pos).max(dim=1)[0]

        max_dist = distance_matrix.max().detach()
        hardest_negative_dist = (distance_matrix + max_dist * mask_pos).min(dim=1)[0]

        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        return triplet_loss.mean()

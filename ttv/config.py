# verf/ttv/config.py
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Dict, Tuple

@dataclass(frozen=True)
class Config:
    seed: int = 42

    # Training
    batch_size: int = 64
    epochs: int = 100
    lr: float = 0.0005
    weight_decay: float = 1e-4

    # Loss
    margin: float = 0.2

    # Data / windowing
    input_channels: int = 6  # <--- UPDATED: Was 9, now 6 (Acc+Gyr)
    window_size: int = 1000  # <--- RESTORED: 5 seconds (200Hz)
    stride: int = 500        # 50% overlap

    # Sensor streams
    streams: Tuple[str, ...] = (
        "Pelvis", "Upper_Spine", "Shank_LT", "Foot_LT", "Shank_RT", "Foot_RT",
    )

    # Model dims
    feat_dim: int = 128
    n_head: int = 8
    num_layers: int = 4
    emb_dim: int = 128

    # Scheduler
    sched_mode: str = "min"
    sched_factor: float = 0.5
    sched_patience: int = 5

CFG = Config()

def make_cfg(**overrides: Any) -> Config:
    return replace(CFG, **overrides)

def build_model_signature(cfg: Config) -> Dict[str, Any]:
    return {
        "model_name": "HybridGaitTransformer",
        "feat_dim": cfg.feat_dim,
        "n_head": cfg.n_head,
        "layers": cfg.num_layers
    }

def build_training_signature(optimizer_type: str, scheduler_type: str) -> Dict[str, Any]:
    return {"optimizer_type": optimizer_type, "scheduler_type": scheduler_type}

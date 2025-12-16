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
    lr: float = 0.0003
    # STRATEGY 1: Increase Weight Decay (Was 1e-4 -> Now 1e-3)
    # This prevents weights from growing too large, reducing overfitting.
    weight_decay: float = 1e-4

    # Loss
    # STRATEGY 2: Increase Margin (Was 0.2 -> Now 0.5)
    # This forces the model to push negatives much further away.
    margin: float = 0.4

    # Data / windowing
    input_channels: int = 6 
    window_size: int = 1000
    stride: int = 500

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

# verf/ttv/config.py
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Dict, Tuple

@dataclass(frozen=True)
class Config:
    # Repro
    seed: int = 42

    # Training
    batch_size: int = 64  # P-K Sampling (e.g., 8 people * 8 samples)
    epochs: int = 100     # Transformers need longer to converge
    lr: float = 0.0005    # Lower LR for Transformer stability
    weight_decay: float = 1e-4

    # Loss
    margin: float = 0.2   # Tighter margin for precise metric learning

    # Data / windowing
    input_channels: int = 9  # 9 axes (Acc+Gyr+Mag) to match your 54-channel spec? (Adjust if 6)
    window_size: int = 500   # 5 seconds @ 100Hz
    stride: int = 250        # 50% overlap

    # Sensor streams (Total channels = len(streams) * input_channels)
    streams: Tuple[str, ...] = (
        "Pelvis", "Upper_Spine", "Shank_LT", "Foot_LT", "Shank_RT", "Foot_RT",
    )

    # Hybrid Transformer Dims
    feat_dim: int = 128      # CNN output channels / Transformer d_model
    n_head: int = 8          # Attention heads
    num_layers: int = 4      # Transformer Encoder layers
    emb_dim: int = 128       # Final biometric fingerprint size

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

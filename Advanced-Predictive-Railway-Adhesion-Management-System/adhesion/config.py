"""Configuration for the DETR-based predictive adhesion model."""
from dataclasses import dataclass, field
from typing import Dict, Tuple

SURFACE_CLASSES = ("ballast_sleepers", "continuous_concrete", "level_crossing", "bridge_open_track", "unknown")
RAIL_CLASSES = ("dry", "wet", "leaves", "contaminated", "frost_ice")
ENV_CLASSES = ("spring", "summer", "autumn", "winter", "unknown")


@dataclass
class ModelConfig:
    # input
    image_size: Tuple[int, int] = (224, 384)  # (H, W)
    n_axles: int = 4
    # visual encoders / fusion (Sec. 2)
    enc_channels: Tuple[int, ...] = (32, 64, 128, 192)  # stride 2 per stage -> total stride 16
    # transformer (Sec. 3)
    d_model: int = 256
    n_heads: int = 8
    n_enc_layers: int = 3
    n_dec_layers: int = 3  # L
    dim_ff: int = 1024
    dropout: float = 0.1
    # query groups (Sec. 3): number of queries per group
    queries_per_group: Dict[str, int] = field(default_factory=lambda: {
        "texture": 4, "continuity": 1, "surface": 2, "rail": 2, "environment": 1, "adhesion": 2})
    # texture representation: [fine, coarse, periodic, irregular, contrast]
    tex_component_dim: int = 8
    n_surface: int = len(SURFACE_CLASSES)
    n_rail: int = len(RAIL_CLASSES)
    n_env: int = len(ENV_CLASSES)
    # temporal (Sec. 4)
    gru_hidden: int = 256
    gru_layers: int = 1
    # dynamics normalisation
    v_scale: float = 40.0        # m/s
    tau_scale: float = 50_000.0  # N m
    # slip control target (Sec. 5)
    slip_target: float = 0.05
    slip_tolerance: float = 0.02
    mu_max: float = 0.6
    # force interface (Sec. 6-7)
    f_max_kn: float = 100.0
    # validity indicator a_t thresholds
    max_sigma_kn: float = 15.0
    min_compat: float = 0.5


@dataclass
class LossWeights:
    tex: float = 0.1
    surface: float = 1.0
    rail: float = 1.0
    env: float = 0.5
    risk: float = 1.0
    force: float = 2.0
    temp: float = 0.1
    # auxiliary terms (not in the formulation's minimal loss; set 0 to disable)
    continuity: float = 0.5
    compat: float = 0.5
    future: float = 0.5
    uncertainty: float = 0.1
    # temporal-loss sensitivity to risk/state change (larger -> penalty vanishes faster when state changes)
    temp_kappa: float = 20.0
    huber_delta: float = 0.1  # in normalised force units (F / Fmax)

"""Datasets. ``SyntheticRailDataset`` produces random-but-consistent sequences purely for wiring/smoke tests;
it is NOT physically meaningful. Replace it with a real dataset returning the same dict keys."""
import torch
from torch.utils.data import Dataset

from .config import ModelConfig


class SyntheticRailDataset(Dataset):
    def __init__(self, cfg: ModelConfig, n_samples: int = 64, seq_len: int = 4, seed: int = 0):
        self.cfg, self.n, self.T, self.seed = cfg, n_samples, seq_len, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        cfg, T = self.cfg, self.T
        g = torch.Generator().manual_seed(self.seed * 100003 + i)
        H, W = cfg.image_size
        # latent "slipperiness" drifts smoothly; labels derive from it so the task is learnable
        slip_lvl = (torch.rand(1, generator=g) + torch.cumsum(0.1 * torch.randn(T, generator=g), 0)).clamp(0, 1)
        rail = torch.where(slip_lvl > 0.5, torch.tensor(1), torch.tensor(0)).long()
        img = torch.rand(T, 3, H, W, generator=g) * (1 - 0.5 * slip_lvl.view(T, 1, 1, 1))
        mask = torch.zeros(T, 1, H, W)
        mask[:, :, H // 2:, W // 3: 2 * W // 3] = 1.0
        speed = 10 + 20 * torch.rand(T, generator=g)
        torque = 20_000 * torch.rand(T, generator=g)
        grade = 0.02 * torch.randn(T, generator=g)
        slip = (0.03 + 0.1 * slip_lvl).unsqueeze(-1).expand(T, cfg.n_axles) + 0.005 * torch.randn(T, cfg.n_axles, generator=g)
        force = cfg.f_max_kn * (0.8 * slip_lvl)
        ones = -torch.ones(T, dtype=torch.long)
        return {
            "images": img, "mask": mask, "slip": slip.clamp_min(0), "speed": speed, "torque": torque, "grade": grade,
            "actuator_ok": torch.ones(T),
            "surface": torch.zeros(T, dtype=torch.long), "rail": rail, "env": ones.clone(),
            "future_rail": torch.cat([rail[1:], rail[-1:]]),
            "continuity": torch.ones(T), "compat": torch.ones(T), "risk": slip_lvl, "force_ref": force,
        }

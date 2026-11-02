"""Multi-task training objective (Sec. 8).

Targets (all optional; missing labels are masked out):
  surface, rail, env      (B,T) long, -1 = unlabeled
  future_rail             (B,T) long, -1 = unlabeled  (rail condition at t+Tp)
  texture                 (B,T,5*tex_component_dim) float + texture_mask (B,T)
  continuity, compat, risk (B,T) float in [0,1], NaN = unlabeled
  force_ref               (B,T) float, kN, NaN = unlabeled
"""
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .config import LossWeights, ModelConfig


def _ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if (target >= 0).sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=-1)


def _masked(fn, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    m = ~torch.isnan(target)
    if m.sum() == 0:
        return pred.sum() * 0.0
    return fn(pred[m], target[m])


def temporal_loss(f_norm: torch.Tensor, risk_ref: torch.Tensor, slip: torch.Tensor, kappa: float) -> torch.Tensor:
    """L_temp = sum_t w_t |F_t - F_{t-1}| with w_t = exp(-kappa * state change).

    State change = |Δ risk| + mean|Δ slip|. Weight vanishes when risk/slip changes, and increases of
    force while the risk is rising are never penalised, so hazards can still trigger rapid increases.
    """
    if f_norm.shape[1] < 2:
        return f_norm.sum() * 0.0
    df = f_norm[:, 1:] - f_norm[:, :-1]
    d_risk = risk_ref[:, 1:] - risk_ref[:, :-1]
    d_risk = torch.nan_to_num(d_risk, nan=0.0)
    d_slip = (slip[:, 1:] - slip[:, :-1]).abs().mean(-1)
    w = torch.exp(-kappa * (d_risk.abs() + d_slip)).detach()
    rising = (d_risk > 0) & (df > 0)
    w = torch.where(rising, torch.zeros_like(w), w)
    return (w * df.abs()).mean()


def compute_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], cfg: ModelConfig,
                 w: LossWeights = LossWeights()) -> Tuple[torch.Tensor, Dict[str, float]]:
    L = {}
    L["surface"] = _ce(out["surface_logits"], batch["surface"])
    L["rail"] = _ce(out["rail_logits"], batch["rail"])
    L["env"] = _ce(out["env_logits"], batch["env"])
    L["future"] = _ce(out["future_rail_logits"], batch["future_rail"])

    if "texture" in batch:
        m = batch["texture_mask"].bool()
        L["tex"] = F.mse_loss(out["z_tex"][m], batch["texture"][m]) if m.any() else out["z_tex"].sum() * 0.0
    else:
        L["tex"] = out["z_tex"].sum() * 0.0

    bce = F.binary_cross_entropy
    L["continuity"] = _masked(bce, out["continuity"].squeeze(-1), batch["continuity"])
    L["compat"] = _masked(bce, out["compat"].squeeze(-1), batch["compat"])
    L["risk"] = _masked(bce, out["risk"].squeeze(-1), batch["risk"])

    f_norm = out["f_raw"].squeeze(-1) / cfg.f_max_kn
    f_ref = batch["force_ref"] / cfg.f_max_kn
    L["force"] = _masked(lambda p, t: F.huber_loss(p, t, delta=w.huber_delta), f_norm, f_ref)

    # Gaussian NLL on (normalised) force; the error is detached so sigma calibrates without distorting the mean.
    logvar = out["f_logvar"].squeeze(-1)
    m = ~torch.isnan(f_ref)
    if m.any():
        err2 = (f_norm[m] - f_ref[m]).detach() ** 2
        L["uncertainty"] = (0.5 * (logvar[m] + err2 * torch.exp(-logvar[m]))).mean()
    else:
        L["uncertainty"] = logvar.sum() * 0.0

    L["temp"] = temporal_loss(f_norm, batch["risk"], batch["slip"], w.temp_kappa)

    total = (w.tex * L["tex"] + w.surface * L["surface"] + w.rail * L["rail"] + w.env * L["env"]
             + w.risk * L["risk"] + w.force * L["force"] + w.temp * L["temp"]
             + w.continuity * L["continuity"] + w.compat * L["compat"] + w.future * L["future"]
             + w.uncertainty * L["uncertainty"])
    logs = {k: float(v.detach()) for k, v in L.items()}
    logs["total"] = float(total.detach())
    return total, logs

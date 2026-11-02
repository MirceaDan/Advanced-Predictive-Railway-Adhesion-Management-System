"""Physical reference model for normal-force augmentation (Sec. 6) and slip error (Sec. 5)."""
import torch

G = 9.81
M_REF_KG = 84_000.0
N_REF_KN = M_REF_KG * G / 1000.0  # ~824.0 kN static weight (NOT automatically the driven-axle load)


def slip_error(s: torch.Tensor, s_target: float, delta: float) -> torch.Tensor:
    """e_t = max(0, |s_t - s*| - delta), element-wise (per axle)."""
    return torch.clamp((s - s_target).abs() - delta, min=0.0)


def ideal_delta_normal_kn(f_traction_req_kn: torch.Tensor, mu: torch.Tensor,
                          n_drive_kn: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """DeltaN_ideal = max(0, F_req / mu - N_drive), Coulomb approximation F_adh,max = mu * N_drive.

    Illustrative only; it must not be used on its own to command the actuator.
    """
    return torch.clamp(f_traction_req_kn / mu.clamp_min(eps) - n_drive_kn, min=0.0)


def command_force(f_raw_kn: torch.Tensor, valid: torch.Tensor, f_max_kn: float) -> torch.Tensor:
    """F_cmd = a_t * sat_[0, Fmax](F_raw)."""
    return valid.to(f_raw_kn.dtype) * f_raw_kn.clamp(0.0, f_max_kn)

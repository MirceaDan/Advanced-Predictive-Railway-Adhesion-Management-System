"""DETR-based predictive railway adhesion network (follows the mathematical formulation, Secs. 2-7)."""
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .physics import command_force, slip_error

GROUPS = ("texture", "continuity", "surface", "rail", "environment", "adhesion")


def _mlp(i: int, h: int, o: int, p: float = 0.0) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, o))


def _conv_stack(in_ch: int, channels) -> nn.Sequential:
    layers = []
    for c in channels:
        layers += [nn.Conv2d(in_ch, c, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c), nn.GELU(),
                   nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.GELU()]
        in_ch = c
    return nn.Sequential(*layers)


def sine_positional_encoding_2d(h: int, w: int, d: int, device) -> torch.Tensor:
    """Fixed 2D sinusoidal P_pos, returned as (h*w, d)."""
    assert d % 4 == 0, "d_model must be divisible by 4"
    dq = d // 4
    freq = torch.exp(torch.arange(dq, device=device) * (-math.log(10000.0) / dq))
    ys = torch.arange(h, device=device).float()[:, None] * freq[None]
    xs = torch.arange(w, device=device).float()[:, None] * freq[None]
    ype = torch.cat([ys.sin(), ys.cos()], -1)[:, None, :].expand(h, w, d // 2)
    xpe = torch.cat([xs.sin(), xs.cos()], -1)[None, :, :].expand(h, w, d // 2)
    return torch.cat([ype, xpe], -1).reshape(h * w, d)


class AdhesionDETR(nn.Module):
    """Inputs are sequences: images (B,T,3,H,W), mask (B,T,1,H,W), slip (B,T,A), v/tau/grade (B,T).

    A single frame is a sequence with T=1; pass the returned ``h`` back as ``h_prev`` for streaming.
    """

    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        d, A = cfg.d_model, cfg.n_axles

        # Sec. 2: phi_I, phi_M, Fuse, projection W_f
        self.enc_image = _conv_stack(3, cfg.enc_channels)
        self.enc_mask = _conv_stack(1, cfg.enc_channels)
        c = cfg.enc_channels[-1]
        # Fuse: concatenation + gated modulation (the BEG overlay says *where* to look)
        self.fuse = nn.Sequential(nn.Conv2d(2 * c, c, 1), nn.GELU())
        self.mask_gate = nn.Conv2d(c, c, 1)
        self.proj = nn.Linear(c, d)  # W_f, b_f

        # Sec. 3: encoder, grouped learnable queries, decoder
        enc_layer = nn.TransformerEncoderLayer(d, cfg.n_heads, cfg.dim_ff, cfg.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, cfg.n_enc_layers, enable_nested_tensor=False)
        dec_layer = nn.TransformerDecoderLayer(d, cfg.n_heads, cfg.dim_ff, cfg.dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, cfg.n_dec_layers, norm=nn.LayerNorm(d))
        counts = [cfg.queries_per_group[g] for g in GROUPS]
        self.q0 = nn.Parameter(torch.randn(sum(counts), d) * 0.02)  # Q^(0)
        bounds = [0]
        for n in counts:
            bounds.append(bounds[-1] + n)
        self.slices = {g: slice(bounds[i], bounds[i + 1]) for i, g in enumerate(GROUPS)}

        # Sec. 3.1-3.3 heads
        tc = cfg.tex_component_dim
        self.f_texture = _mlp(d, d, 5 * tc)  # [fine, coarse, periodic, irregular, contrast]
        self.f_continuity = _mlp(d, d // 2, 1)
        self.f_surface = _mlp(d, d // 2, cfg.n_surface)
        self.f_compat = _mlp(5 * tc + 1 + cfg.n_surface, d // 2, 1)
        self.f_rail = _mlp(d, d // 2, cfg.n_rail)
        self.f_env = _mlp(d, d // 2, cfg.n_env)

        # Sec. 4: temporal model, input [Pool(Q); s; v; tau; gamma]
        self.gru = nn.GRU(d + A + 3, cfg.gru_hidden, cfg.gru_layers, batch_first=True)
        self.f_future = _mlp(cfg.gru_hidden, cfg.gru_hidden // 2, cfg.n_rail)  # rail condition at t + Tp

        # Sec. 5: risk, demand, effective adhesion
        H = cfg.gru_hidden
        self.f_risk = _mlp(H + 5 * tc + cfg.n_rail + cfg.n_surface + d, H // 2, 1)
        self.f_demand = _mlp(H + 1 + A + 3, H // 2, 1)
        self.f_mu = _mlp(H + A + 2, H // 2, 2)  # mean and log-variance

        # Sec. 7: force head u_t -> (raw logit, log-variance)
        u_dim = H + 5 * tc + 1 + 1 + cfg.n_rail + cfg.n_env + 1 + A + 3
        self.f_force = _mlp(u_dim, H, 2)
        self._u_dim = u_dim

    # ---- Sec. 2 ----
    def encode_visual(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        ei, em = self.enc_image(img), self.enc_mask(mask)
        z0 = self.fuse(torch.cat([ei, ei * torch.sigmoid(self.mask_gate(em)) + em], 1))
        B, C, h, w = z0.shape
        z = self.proj(z0.flatten(2).transpose(1, 2))  # (B, hw, d)
        return z + sine_positional_encoding_2d(h, w, self.cfg.d_model, z.device)

    def forward(self, images, mask, slip, speed, torque, grade,
                h_prev: Optional[torch.Tensor] = None,
                actuator_ok: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        B, T = images.shape[:2]

        # vision on flattened frames
        z = self.encode_visual(images.flatten(0, 1), mask.flatten(0, 1))
        z = self.encoder(z)
        q = self.decoder(self.q0.unsqueeze(0).expand(z.shape[0], -1, -1), z)  # (BT, Nq, d)
        g = {k: q[:, s].mean(1).view(B, T, -1) for k, s in self.slices.items()}
        pool_q = q.mean(1).view(B, T, -1)

        # heads on group queries
        z_tex = self.f_texture(g["texture"])
        c_hat = torch.sigmoid(self.f_continuity(g["continuity"]))
        surface_logits = self.f_surface(g["surface"])
        p_surface = surface_logits.softmax(-1)
        a_surface = torch.sigmoid(self.f_compat(torch.cat([z_tex, c_hat, p_surface], -1)))
        rail_logits = self.f_rail(g["rail"])
        p_rail = rail_logits.softmax(-1)
        env_logits = self.f_env(g["environment"])
        p_env = env_logits.softmax(-1)

        # dynamics
        v = (speed / cfg.v_scale).unsqueeze(-1)
        tau = (torque / cfg.tau_scale).unsqueeze(-1)
        gam = grade.unsqueeze(-1)
        dyn = torch.cat([slip, v, tau, gam], -1)

        # Sec. 4
        h_seq, h_last = self.gru(torch.cat([pool_q, dyn], -1), h_prev)
        future_rail_logits = self.f_future(h_seq)

        # Sec. 5
        r_hat = torch.sigmoid(self.f_risk(torch.cat([h_seq, z_tex, p_rail, p_surface, g["adhesion"]], -1)))
        e = slip_error(slip, cfg.slip_target, cfg.slip_tolerance)
        demand = F.softplus(self.f_demand(torch.cat([h_seq, r_hat, e, v, tau, gam], -1)))
        mu_out = self.f_mu(torch.cat([h_seq, e, tau, v], -1))
        mu_hat = cfg.mu_max * torch.sigmoid(mu_out[..., :1])
        mu_logvar = mu_out[..., 1:].clamp(-8, 4)  # retained uncertainty on mu

        # Sec. 7
        u = torch.cat([h_seq, z_tex, c_hat, a_surface, p_rail, p_env, r_hat, dyn], -1)
        fo = self.f_force(u)
        f_raw = cfg.f_max_kn * torch.sigmoid(fo[..., :1])  # kN, in [0, Fmax]
        f_logvar = fo[..., 1:].clamp(-8, 4)
        # sigma in kN: normalised-units std scaled by Fmax
        sigma_f = cfg.f_max_kn * torch.exp(0.5 * f_logvar)

        valid = (sigma_f < cfg.max_sigma_kn) & (a_surface > cfg.min_compat)
        if actuator_ok is not None:
            valid = valid & actuator_ok.unsqueeze(-1).bool()
        f_cmd = command_force(f_raw, valid, cfg.f_max_kn)

        return {
            "z_tex": z_tex, "continuity": c_hat, "surface_logits": surface_logits, "p_surface": p_surface,
            "compat": a_surface, "rail_logits": rail_logits, "p_rail": p_rail,
            "env_logits": env_logits, "p_env": p_env,
            "future_rail_logits": future_rail_logits,
            "risk": r_hat, "slip_error": e, "demand": demand, "mu": mu_hat, "mu_logvar": mu_logvar,
            "f_raw": f_raw, "f_logvar": f_logvar, "sigma_f": sigma_f,
            "valid": valid.float(), "f_cmd": f_cmd,  # Y_t = [F_cmd, sigma_F, a_t]
            "h": h_last,
        }

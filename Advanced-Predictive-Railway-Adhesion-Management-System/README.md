# Advanced Predictive Railway Adhesion Management System

PyTorch implementation of the DETR-based predictive adhesion model specified in
[`detr_predictive_railway_adhesion.pdf`](detr_predictive_railway_adhesion.pdf). From a camera image, a BEG track-geometry
overlay and locomotive dynamics, the network requests a **normal-force augmentation in kN**
(`Y_t = [F_cmd, σ_F, a_t]`).

> **Status:** code only. No training has been run, and no dataset or reference-force labels are bundled.
> The output is a research prototype and requires physically justified supervision and separate safety validation
> before any use on a vehicle.

## Layout

```
adhesion/
  config.py   ModelConfig / LossWeights, class vocabularies
  physics.py  slip error e_t, ΔN_ideal, F_cmd = a_t·sat(F_raw), reference weight (84 t -> ~824 kN)
  model.py    AdhesionDETR (Secs. 2-7)
  losses.py   multi-task loss (Sec. 8) with label masking
  data.py     SyntheticRailDataset (wiring/smoke tests only)
  train.py    fit() + CLI
notebooks/training.ipynb   end-to-end training walkthrough
```

## Formulation → code

| Section | Implementation |
|---|---|
| 2 Visual fusion | `enc_image` (φ_I), `enc_mask` (φ_M), `fuse` (BEG-gated fusion), `proj` (W_f, b_f) + 2D sine `P_pos` |
| 3 DETR | `TransformerEncoder`, learnable queries `q0` split into groups (texture, continuity, surface, rail, environment, adhesion), `TransformerDecoder` (L layers) |
| 3.1-3.3 Heads | `f_texture` → `[fine, coarse, periodic, irregular, contrast]`; `f_continuity` (σ); `f_surface`, `f_rail`, `f_env` (softmax); `f_compat` |
| 4 Temporal | GRU over `[Pool(Q); s; v; τ; γ]`; `f_future` predicts rail condition at `t+Tp` |
| 5 Adhesion demand | `f_risk` (σ, risk ∈ [0,1]), `slip_error` (`max(0,|s−s*|−δ)`), `f_demand` (softplus), `f_mu` (mean + log-variance) |
| 7 Force head | `F_raw = Fmax·σ(f_F(u))`, uncertainty `σ_F`, validity `a_t`, `F_cmd = a_t·sat[0,Fmax](F_raw)` |
| 8 Loss | CE (surface/rail/env), Huber (force), temporal, risk, texture; plus optional continuity/compat/future/NLL terms |

Design choices where the formulation leaves freedom:

- `a_t` is rule-based: `σ_F < max_sigma_kn`, `compat > min_compat`, and an optional external `actuator_ok` flag.
- `L_temp` is `|F_t − F_{t−1}|` weighted by `exp(−κ(|Δrisk| + mean|Δslip|))`, and zeroed when risk and force
  rise together, so hazards can still cause rapid force increases.
- `σ_F` comes from a Gaussian-NLL head trained on the detached force error.
- Dynamic inputs are always required by the model; for a vision-only variant feed zeros.
- Slip may be per-axle (`n_axles` in `ModelConfig`); a scalar slip can be passed with `n_axles=1`.

## Usage

```bash
pip install -r requirements.txt
```

```python
import torch
from adhesion import AdhesionDETR, ModelConfig

model = AdhesionDETR(ModelConfig()).eval()
out = model(images, mask, slip, speed, torque, grade)          # (B,T,3,H,W), (B,T,1,H,W), (B,T,A), (B,T) x3
f_cmd, sigma, valid = out["f_cmd"], out["sigma_f"], out["valid"]   # kN
# streaming: model(..., h_prev=out["h"]) with T=1
```

### Data contract

Each sample is a sequence of length `T` (keys → shape): `images (T,3,H,W)`, `mask (T,1,H,W)`, `slip (T,A)`,
`speed`, `torque`, `grade`, `actuator_ok` `(T)`, and labels `surface`, `rail`, `env`, `future_rail` (long, `-1` =
unlabeled), `continuity`, `compat`, `risk`, `force_ref` (float, `NaN` = unlabeled, `force_ref` in kN), optionally
`texture (T,5·tex_component_dim)` with `texture_mask (T)`. See `adhesion/losses.py`.

### Training (when you are ready)

```bash
python -m adhesion.train --synthetic --epochs 2     # wiring smoke test on random data
```

For real data, implement a `Dataset` with the contract above and call `adhesion.train.fit(model, train_ds, val_ds, ...)`,
or follow `notebooks/training.ipynb`.

## Limitations

Outputs are visual risk estimates, not measurements of friction, wheel slip, negative pressure, or airflow.
The Coulomb relation in `physics.ideal_delta_normal_kn` is illustrative and is not used to command the actuator.

Explained Like-I'm-5:
RGB (3ch) ──► CNN enc ─┐
BEG mask (1ch) ► CNN enc ─┴► gated fuse (stride 16) ► Linear→d=256 + 2D sine pos
   ► Transformer encoder (3 layers)
   ► Transformer decoder (3 layers), 12 learnable queries:
        texture 4 | continuity 1 | surface 2 | rail 2 | environment 1 | adhesion 2
   ► each group is mean-pooled into one 256-d vector and goes to its own MLP head

Per frame, the heads produce:

• Texture: a 40-d latent.
• Continuity: a sigmoid score.
• Surface: 5 class probabilities.
• Compatibility: an MLP over  [texture, continuity, surface] .
• Rail: 5 class probabilities.
• Environment: 5 class probabilities.

The GRU (hidden size 256) consumes  [mean of all queries; slip(4 axles); v/40; τ/50k; grade] .

Inputs and outputs

Inputs, as sequences:

• Image  (B,T,3,H,W)  and BEG mask  (B,T,1,H,W) .
• Per-axle slip  (B,T,4) .
• Speed, torque and grade  (B,T) .
• The previous GRU state.
• An optional  actuator_ok  flag.

Outputs:

• Force interface:  F_cmd  in kN,  σ_F  in kN, and the validity flag  a_t .
• Auxiliary outputs: risk ∈ [0,1], demand, μ with its log-variance, the class distributions, future rail condition, and the next GRU state.

How the heads feed the kN prediction

The heads do not carry hand-set weights. They are concatenated into one vector, and a single 2-layer MLP learns how much each matters:

u = [h (256) | z_tex (40) | continuity | compat | p_rail (5) | p_env (5) | risk | slip(4), v, τ, grade]
F_raw = Fmax · sigmoid(MLP(u)),  Fmax = 100 kN

• Training signal: the force head is trained only against  force_ref , using Huber loss. The other heads shape  u  through their own losses, and through gradients from the force loss, which are not detached.
• Loss weights: the  λ s (force 2.0, surface 1.0, risk 1.0 and so on) only balance training. They do not set how much each head contributes at inference.
• Gating: only the validity flag is rule-based, not learned.  a_t=1  requires  σ_F < 15 kN ,  compat > 0.5  and  actuator_ok . Otherwise  F_cmd = 0 .
• Not used for the output:
• The demand and μ heads are computed but never feed into  u .
•  physics.ideal_delta_normal_kn  (F/μ − N) is not wired in. The network does not compute the physical ΔN.
• No supervised link: nothing forces the network to respect  F/μ − N  or any adhesion physics. It is a black-box regressor on whatever  force_ref  labels you supply.

Until you have trained it on real labels, every output is meaningless.

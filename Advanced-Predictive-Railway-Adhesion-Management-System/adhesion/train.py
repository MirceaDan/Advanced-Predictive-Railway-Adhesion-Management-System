"""Training loop. Importable (`fit`) and runnable: `python -m adhesion.train --synthetic` (smoke test only)."""
import argparse
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .config import LossWeights, ModelConfig
from .data import SyntheticRailDataset
from .losses import compute_loss
from .model import AdhesionDETR

INPUT_KEYS = ("images", "mask", "slip", "speed", "torque", "grade")


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def run_epoch(model, loader, cfg, weights, device, optimizer=None, scaler=None, grad_clip=1.0):
    train = optimizer is not None
    model.train(train)
    sums, n = {}, 0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.set_grad_enabled(train), torch.autocast(device.type, enabled=scaler is not None):
            out = model(*[batch[k] for k in INPUT_KEYS], actuator_ok=batch.get("actuator_ok"))
        loss, logs = compute_loss({k: v.float() for k, v in out.items()}, batch, cfg, weights)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        for k, v in logs.items():
            sums[k] = sums.get(k, 0.0) + v
        n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}


def fit(model: AdhesionDETR, train_ds: Dataset, val_ds: Optional[Dataset] = None, epochs: int = 20,
        batch_size: int = 4, lr: float = 1e-4, weight_decay: float = 1e-4, weights: LossWeights = LossWeights(),
        device: Optional[str] = None, ckpt_dir: str = "checkpoints", num_workers: int = 0, amp: bool = True):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    cfg = model.cfg
    tl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    vl = DataLoader(val_ds, batch_size, num_workers=num_workers) if val_ds else None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler() if (amp and device.type == "cuda") else None
    os.makedirs(ckpt_dir, exist_ok=True)
    best, history = float("inf"), []
    for ep in range(1, epochs + 1):
        tr = run_epoch(model, tl, cfg, weights, device, opt, scaler)
        va = run_epoch(model, vl, cfg, weights, device) if vl else None
        sched.step()
        history.append({"epoch": ep, "train": tr, "val": va})
        print(f"epoch {ep:03d} train {tr['total']:.4f}" + (f" val {va['total']:.4f}" if va else ""))
        score = (va or tr)["total"]
        state = {"model": model.state_dict(), "cfg": cfg, "epoch": ep}
        torch.save(state, os.path.join(ckpt_dir, "last.pt"))
        if score < best:
            best = score
            torch.save(state, os.path.join(ckpt_dir, "best.pt"))
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true", help="use the synthetic wiring-test dataset")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seq-len", type=int, default=4)
    p.add_argument("--ckpt-dir", default="checkpoints")
    a = p.parse_args()
    if not a.synthetic:
        raise SystemExit("No real dataset is bundled. Implement a Dataset returning the keys documented in "
                         "adhesion/losses.py and call adhesion.train.fit(), or pass --synthetic for a smoke test.")
    cfg = ModelConfig()
    fit(AdhesionDETR(cfg), SyntheticRailDataset(cfg, 64, a.seq_len), SyntheticRailDataset(cfg, 16, a.seq_len, seed=1),
        a.epochs, a.batch_size, a.lr, ckpt_dir=a.ckpt_dir)


if __name__ == "__main__":
    main()

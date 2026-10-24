"""Optimizer / scheduler construction. Owner: Member C (PR-20)."""
import math
from typing import Dict

import torch


def build_optimizer(model, cfg: Dict) -> torch.optim.Optimizer:
    """Optimizer with one parameter group per `model.param_groups(...)` entry: a small LR for the
    pretrained encoder (`lr_encoder`) and a larger one for the new modules (`lr_new`).

    cfg['train']['optimizer']: name (adam | adamw), lr_encoder, lr_new, weight_decay. 'adam' is the
    baseline's choice (L2 weight decay inside Adam); 'adamw' decouples the decay.
    """
    opt_cfg = cfg["train"]["optimizer"]
    name = str(opt_cfg.get("name", "adam")).lower()
    groups = model.param_groups(
        float(opt_cfg["lr_encoder"]), float(opt_cfg["lr_new"]), float(opt_cfg.get("weight_decay", 0.0))
    )
    if name == "adam":
        return torch.optim.Adam(groups)
    if name == "adamw":
        return torch.optim.AdamW(groups)
    raise ValueError(f"train.optimizer.name must be adam or adamw, got {name!r}")


def lr_factor(step: int, warmup_steps: int, total_steps: int, name: str = "cosine") -> float:
    """Multiplier on every group's base LR after `step` optimizer steps.
    Linear warmup (first step already > 0), then cosine decay to 0 (or constant)."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if name == "constant":
        return 1.0
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max(step - warmup_steps, 0) / decay_steps, 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_scheduler(optimizer, cfg: Dict, steps_per_epoch: int):
    """Per-step LambdaLR: warmup for `warmup_epochs`, then cosine over the remaining epochs.
    Call `scheduler.step()` after every optimizer step (the Trainer does)."""
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    train_cfg = cfg["train"]
    sched = train_cfg.get("scheduler") or {}
    name = str(sched.get("name", "cosine")).lower()
    if name not in ("cosine", "constant"):
        raise ValueError(f"train.scheduler.name must be cosine or constant, got {name!r}")
    warmup = int(round(float(sched.get("warmup_epochs", 0)) * steps_per_epoch))
    total = int(train_cfg["epochs"]) * steps_per_epoch
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, warmup, total, name))

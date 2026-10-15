"""Training loop. Owner: Member C (step C6).
Uses evaluation.evaluate (Member A) for the per-epoch dev EER used by early stopping."""
from typing import Dict


class Trainer:
    def __init__(self, cfg: Dict, model, train_loader, dev_loader):
        raise NotImplementedError("TODO(C6)")

    def fit(self) -> str:
        """Run training; return path of best checkpoint (by dev EER).
        Must support: AMP (STFT/ULCP stay fp32), grad clipping, resume, deterministic seeding,
        and a `overfit_batch` debug mode that trains on one batch to ~0 loss (step C6 sanity test)."""
        raise NotImplementedError("TODO(C6)")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        raise NotImplementedError("TODO(C6)")

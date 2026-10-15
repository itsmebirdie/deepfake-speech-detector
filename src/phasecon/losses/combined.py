"""Eq. 6: L = L_bin + beta * L_voc + mu * L_cons. Owner: Member C (step C5)."""
from typing import Dict, Tuple

import torch


def total_loss(out: Dict[str, torch.Tensor], batch: Dict, beta: float, mu: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Returns (loss, parts) where parts = {'bin':..., 'voc':..., 'cons':...} as floats for logging.
    Skips terms whose keys are absent from `out` (baseline-only / no-phase configs)."""
    raise NotImplementedError("TODO(C5)")

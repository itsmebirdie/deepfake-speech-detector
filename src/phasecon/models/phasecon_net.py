"""Full model. Owner: Member C (step C4). Contract: docs/INTERFACES.md section 5."""
from typing import Dict

import torch
import torch.nn as nn


class PhaseConNet(nn.Module):
    """forward(wav [B,T]) -> dict:
        'logits_bin':      [B,2]
        'logits_voc':      [B,n_voc]            (if vocoder head enabled)
        'cons_per_sample': [B]                  (from PhaseBranch; absent if phase branch disabled)
    With cfg model.phase_branch.enabled=false this must reduce to the baseline (used to validate the pipeline)."""

    def __init__(self, cfg: Dict):
        super().__init__()
        raise NotImplementedError("TODO(C4)")

    def forward(self, wav: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("TODO(C4)")

    def param_groups(self, lr_encoder: float, lr_new: float, weight_decay: float):
        """Separate LR for pretrained encoder vs new modules (used by training/optim.py)."""
        raise NotImplementedError("TODO(C6)")

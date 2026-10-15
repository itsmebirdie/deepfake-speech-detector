"""Phase-discrepancy stream = STFT -> ULCP -> discrepancy tensors -> encoder. Owner: Member B (B7).
THIS IS THE B -> C HAND-OFF INTERFACE. Do not change the output keys without telling Member C."""
from typing import Dict

import torch
import torch.nn as nn


class PhaseBranch(nn.Module):
    """forward(wav [B,T]) -> dict:
        'H_d':            [B, T_d, C_d]   features for fusion
        'cons_per_sample':[B]             mean_{r,f,t} w * (1 - cos dphi)   (used by L_cons, masked to bona fide)
        'delta_phi':      list of [B,F_r,N_r]  (debug / visualisation only; may be omitted in training)
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        raise NotImplementedError("TODO(B7)")

    def forward(self, wav: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("TODO(B7)")

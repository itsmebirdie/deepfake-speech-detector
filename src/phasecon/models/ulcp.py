"""Unrolled Learnable Consistency Projection (Eqs. 1-3). Owner: Member B (step B4)."""
from typing import List

import torch
import torch.nn as nn

from phasecon.signal.stft import MultiResSTFT


class ULCPStep(nn.Module):
    """One unrolled iteration k: Z=S(S^-1(Y)); W=Z+alpha_k(Z-Z_prev)+g_theta_k(Z); Y'=A*W/(|W|+eps)."""

    def __init__(self, momentum_init: float, learnable_momentum: bool, correction_channels: int):
        super().__init__()
        raise NotImplementedError("TODO(B4)")

    def forward(self, Y, Z_prev, A, stft: MultiResSTFT, r: int, length: int):
        """Returns (Y_next, Z). g_theta acts on stacked [Re Z, Im Z] and returns a complex-compatible correction."""
        raise NotImplementedError("TODO(B4)")


class ULCP(nn.Module):
    """forward(A [B,F,N] magnitude, stft, r, length) -> Y_K complex [B,F,N] with |Y_K| == A.
    phi_hat = angle(Y_K). Deterministic init (no random phase). K steps from config."""

    def __init__(self, num_steps: int, init: str, momentum_init: float,
                 learnable_momentum: bool, correction_channels: int, learnable: bool = True):
        super().__init__()
        raise NotImplementedError("TODO(B4)")

    def forward(self, A: torch.Tensor, stft: MultiResSTFT, r: int, length: int) -> torch.Tensor:
        raise NotImplementedError("TODO(B4)")

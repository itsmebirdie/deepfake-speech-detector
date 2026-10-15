"""Differentiable multi-resolution STFT / iSTFT. Owner: Member B (step B1)."""
from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class STFTConfig:
    n_fft: int
    hop: int
    win_length: int
    center: bool = True


class MultiResSTFT(nn.Module):
    """forward(wav [B,T]) -> list of complex tensors, one per resolution, each [B, F_r, N_r]
    with F_r = n_fft//2 + 1. Always computed in float32 (disable autocast inside).
    inverse(X [B,F,N], r, length) -> wav [B,length]
    """

    def __init__(self, resolutions: Sequence[STFTConfig]):
        super().__init__()
        raise NotImplementedError("TODO(B1)")

    def forward(self, wav: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError("TODO(B1)")

    def inverse(self, X: torch.Tensor, r: int, length: int) -> torch.Tensor:
        raise NotImplementedError("TODO(B1)")

    def project(self, X: torch.Tensor, r: int, length: int) -> torch.Tensor:
        """S(S^-1(X)): projection onto the set of consistent STFTs (Eq. 1)."""
        raise NotImplementedError("TODO(B1)")

"""Phase residual with FIXED Griffin-Lim, for the feasibility study (PR-10). Owner: Member B (step B3).

The project's idea in one function: keep a clip's magnitude A = |X|, let fast Griffin-Lim rebuild a
phase Y from A alone, and measure how far the clip's real phase is from that rebuilt phase:

    dphi = angle(X * conj(Y))              phase difference per (bin, frame); conj = Eq. 4 fix, ARCHITECTURE.md
    w    = A / (A + tau)                   confidence weight: quiet bins (noise) count little
    map  = w * (1 - cos(dphi))             0 = same phase, 2 = opposite phase

`1 - cos` instead of |dphi| so that a phase off by a full turn (2*pi) counts as no difference.
PR-16 builds the learnable model's version of this (cos/sin channels + IF); this is the fixed reference.
"""
import math

import torch

from .griffin_lim import fast_griffin_lim
from .stft import MultiResSTFT

INITS = ("zeros", "linear_group_delay")


def initial_phase(shape, n_fft: int, kind: str, device=None) -> torch.Tensor:
    """Deterministic starting phase for Griffin-Lim, shape [..., F, N].

    zeros               every bin starts at phase 0
    linear_group_delay  bin k starts at -pi*k (a constant delay of half a frame), a common phase-retrieval start
    """
    if kind == "zeros":
        return torch.zeros(shape, device=device)
    if kind == "linear_group_delay":
        n_bins = shape[-2]
        k = torch.arange(n_bins, device=device, dtype=torch.float32).unsqueeze(-1)
        return (-math.pi * k * (n_fft // 2) * 2 / n_fft).expand(shape).clone()
    raise ValueError(f"unknown init {kind!r}; expected one of {INITS}")


def phase_residual_map(
    wav: torch.Tensor, stft: MultiResSTFT, r: int, n_iter: int = 32, init: str = "zeros",
    tau: float = 1e-3, momentum: float = 0.99,
) -> torch.Tensor:
    """wav [B,T] -> confidence-weighted residual map w * (1 - cos dphi), shape [B, F, N], values in [0, 2)."""
    X = stft.stft(wav, r)
    A = X.abs()
    phi0 = initial_phase(A.shape, stft.configs[r].n_fft, init, device=A.device)
    Y = fast_griffin_lim(A, stft, r, wav.shape[-1], n_iter=n_iter, momentum=momentum, init_phase=phi0)
    dphi = torch.angle(X * torch.conj(Y))
    w = A / (A + tau)
    return w * (1 - torch.cos(dphi))


def clip_score(residual_map: torch.Tensor) -> torch.Tensor:
    """[B, F, N] -> [B]: the clip-level mean of the residual map (PR-10's separability statistic)."""
    return residual_map.mean(dim=(-2, -1))

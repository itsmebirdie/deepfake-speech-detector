"""Classical fast Griffin-Lim (non-learnable reference). Owner: Member B (step B2).
Used for (a) the feasibility sanity check, (b) ablation (ii), (c) unit-testing ULCP.

Griffin-Lim recovers a phase that fits a given magnitude: start from some phase, then repeat
  1. make the STFT consistent: go to a waveform and back (MultiResSTFT.project),
  2. put the target magnitude back, keep the new phase.
The "fast" variant (Perraudin, Balazs & Sondergaard, 2013) adds momentum: each step also moves
further in the direction of the last change, which converges in far fewer iterations.
"""
import torch

from .stft import MultiResSTFT


def _with_magnitude(X: torch.Tensor, mag: torch.Tensor) -> torch.Tensor:
    """Keep X's phase, replace its magnitude by `mag` (a zero bin keeps phase 0)."""
    return torch.polar(mag, torch.angle(X))


def fast_griffin_lim(
    mag: torch.Tensor,
    stft: MultiResSTFT,
    r: int,
    length: int,
    n_iter: int = 32,
    momentum: float = 0.99,
    init_phase: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return complex STFT estimate [B,F,N] whose magnitude is `mag` (Griffin & Lim 1984; fast variant).

    mag [B,F,N] (real, >= 0) at resolution r; `init_phase` [B,F,N] defaults to zeros (deterministic).
    momentum=0 gives the classic Griffin-Lim. Differentiable; runs in float32.
    """
    if n_iter < 0:
        raise ValueError(f"n_iter must be >= 0, got {n_iter}")
    mag = mag.float()
    phase = torch.zeros_like(mag) if init_phase is None else init_phase.float()
    if phase.shape != mag.shape:
        raise ValueError(f"init_phase {tuple(phase.shape)} does not match mag {tuple(mag.shape)}")
    estimate = torch.polar(mag, phase)
    previous = estimate
    for _ in range(n_iter):
        current = _with_magnitude(stft.project(estimate, r, length), mag)
        estimate = current + momentum * (current - previous)
        previous = current
    return previous


def spectral_convergence(Y: torch.Tensor, mag: torch.Tensor, stft: MultiResSTFT, r: int, length: int) -> torch.Tensor:
    """|| |S(S^-1(Y))| - mag || / || mag ||: how far Y is from being a consistent STFT with magnitude mag."""
    rebuilt = stft.project(Y, r, length).abs()
    return torch.linalg.vector_norm(rebuilt - mag) / torch.linalg.vector_norm(mag)

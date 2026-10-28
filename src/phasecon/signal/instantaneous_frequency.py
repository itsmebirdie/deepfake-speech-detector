"""Instantaneous-frequency utilities. Owner: Member B (step B2).

Phase is only defined up to a multiple of 2*pi, so every difference of phases goes through
`wrap_to_pi` (or cos/sin), never a plain subtraction.
"""
import math

import torch


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """Wrap angles to (-pi, pi]."""
    return math.pi - torch.remainder(math.pi - x, 2 * math.pi)


def instantaneous_frequency(phase: torch.Tensor, hop: int, n_fft: int) -> torch.Tensor:
    """Time-derivative of unwrapped phase (principal-value difference, hop/frequency-bin corrected).
    phase [B,F,N] -> [B,F,N] (first frame zero-padded).

    Units: radians per sample, so frequency in Hz = IF * sample_rate / (2*pi).
    Method: between two frames a component exactly at bin k's centre frequency (2*pi*k/n_fft rad/sample)
    advances its phase by 2*pi*k*hop/n_fft. The measured advance minus that expected advance, wrapped
    to (-pi, pi], is the deviation from the bin centre; IF = (expected advance + deviation) / hop.
    Exact as long as the true frequency is within n_fft / (2*hop) bins of the bin centre.
    """
    if phase.dim() < 2:
        raise ValueError(f"expected phase [..., F, N], got shape {tuple(phase.shape)}")
    n_bins = phase.shape[-2]
    if n_bins != n_fft // 2 + 1:
        raise ValueError(f"phase has {n_bins} frequency bins, but n_fft={n_fft} gives {n_fft // 2 + 1}")
    k = torch.arange(n_bins, device=phase.device, dtype=phase.dtype).unsqueeze(-1)  # [F, 1]
    expected = 2 * math.pi * k * hop / n_fft  # phase advance per hop at each bin centre
    advance = phase[..., 1:] - phase[..., :-1]
    deviation = wrap_to_pi(advance - expected)
    inst = (expected + deviation) / hop
    return torch.cat([torch.zeros_like(phase[..., :1]), inst], dim=-1)

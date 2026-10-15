"""Instantaneous-frequency utilities. Owner: Member B (step B2)."""
import torch


def instantaneous_frequency(phase: torch.Tensor, hop: int, n_fft: int) -> torch.Tensor:
    """Time-derivative of unwrapped phase (principal-value difference, hop/frequency-bin corrected).
    phase [B,F,N] -> [B,F,N] (first frame zero-padded)."""
    raise NotImplementedError("TODO(B2)")


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """Wrap angles to (-pi, pi]."""
    raise NotImplementedError("TODO(B2)")

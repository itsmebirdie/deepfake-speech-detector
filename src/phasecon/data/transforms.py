"""Waveform transforms. Owner: Member A (step A3)."""
import torch


def pad_or_crop(wav: torch.Tensor, num_samples: int, random_crop: bool) -> torch.Tensor:
    """Return a [num_samples] tensor: repeat-pad if short, crop (random if training, else start) if long."""
    raise NotImplementedError("TODO(A3)")


def peak_normalize(wav: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Optional peak normalisation. Must match what the baseline expects (check third_party)."""
    raise NotImplementedError("TODO(A3)")

"""Classical fast Griffin-Lim (non-learnable reference). Owner: Member B (step B2).
Used for (a) the feasibility sanity check, (b) ablation (ii), (c) unit-testing ULCP."""
import torch

from .stft import MultiResSTFT


def fast_griffin_lim(
    mag: torch.Tensor,
    stft: MultiResSTFT,
    r: int,
    length: int,
    n_iter: int = 32,
    momentum: float = 0.99,
    init_phase: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return complex STFT estimate [B,F,N] whose magnitude is `mag` (Griffin & Lim 1984; fast variant)."""
    raise NotImplementedError("TODO(B2)")

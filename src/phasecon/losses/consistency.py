"""L_cons. Owner: Member B (step B8).
Penalises the ULCP discrepancy on bona fide samples only, so the projection becomes accurate on real speech.

Why only real clips: ULCP should learn what the phase of REAL speech looks like, given its magnitude. Then on a
fake clip it predicts "real-looking" phase, and the gap to the fake's actual phase (the discrepancy tensor) grows.
If fakes were included, ULCP would also learn to reproduce vocoder phase and the gap would shrink.
"""
import torch


def consistency_loss(cons_per_sample: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """mean of cons_per_sample over samples with label == 0. Return 0 (with grad graph intact) if the
    batch has no bona fide sample. cons_per_sample [B], label [B]."""
    if cons_per_sample.shape != label.shape:
        raise ValueError(f"cons_per_sample {tuple(cons_per_sample.shape)} and label {tuple(label.shape)} differ")
    real = (label == 0).to(cons_per_sample.dtype)
    # masked mean: sum over real clips / number of real clips. With no real clip both are 0; clamping the count
    # to 1 gives 0 * cons (a zero that stays connected to the graph) instead of 0/0 = NaN.
    return (cons_per_sample * real).sum() / real.sum().clamp(min=1.0)

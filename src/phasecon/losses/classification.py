"""L_bin and L_voc. Owner: Member C (step C5)."""
import torch


def binary_loss(logits: torch.Tensor, label: torch.Tensor, class_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Cross-entropy on [B,2] logits. Use the same class weighting as the baseline if it uses one."""
    raise NotImplementedError("TODO(C5)")


def vocoder_loss(logits: torch.Tensor, vocoder_id: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over bona fide + vocoder classes."""
    raise NotImplementedError("TODO(C5)")

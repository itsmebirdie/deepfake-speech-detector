"""L_bin and L_voc. Owner: Member C (PR-15)."""
import torch
import torch.nn.functional as F


def _check(logits: torch.Tensor, target: torch.Tensor, num_classes: int | None = None) -> None:
    if logits.dim() != 2:
        raise ValueError(f"logits must be [B, C], got {tuple(logits.shape)}")
    if num_classes is not None and logits.shape[1] != num_classes:
        raise ValueError(f"logits must have {num_classes} columns, got {logits.shape[1]}")
    if target.shape != (logits.shape[0],):
        raise ValueError(f"target must be [B] = [{logits.shape[0]}], got {tuple(target.shape)}")


def binary_loss(logits: torch.Tensor, label: torch.Tensor, class_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Cross-entropy on [B,2] logits in repo order [bona fide, synthetic], label 0 / 1.

    The baseline uses plain, unweighted cross-entropy (its class weights are commented out in
    main.py), so `class_weights=None` is the faithful default. Computed in float32 even under AMP.
    Log-probabilities are accepted as logits too (log-softmax is idempotent).
    """
    _check(logits, label, 2)
    weight = None if class_weights is None else class_weights.to(device=logits.device, dtype=torch.float32)
    return F.cross_entropy(logits.float(), label.long(), weight=weight)


def vocoder_loss(logits: torch.Tensor, vocoder_id: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over bona fide (0) + vocoder (1..6) classes. float32."""
    _check(logits, vocoder_id)
    return F.cross_entropy(logits.float(), vocoder_id.long())

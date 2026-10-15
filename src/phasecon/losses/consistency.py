"""L_cons. Owner: Member B (step B8).
Penalises the ULCP discrepancy on bona fide samples only, so the projection becomes accurate on real speech."""
import torch


def consistency_loss(cons_per_sample: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """mean of cons_per_sample over samples with label == 0. Return 0 (with grad graph intact) if the
    batch has no bona fide sample. cons_per_sample [B], label [B]."""
    raise NotImplementedError("TODO(B8)")

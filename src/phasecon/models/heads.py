"""Classification heads. Owner: Member C (step C2)."""
import torch
import torch.nn as nn


class GradReverse(torch.autograd.Function):
    """Gradient reversal layer (Ganin & Lempitsky 2015). Ablation (v)."""

    @staticmethod
    def forward(ctx, x, lambd):
        raise NotImplementedError("TODO(C2)")

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("TODO(C2)")


class BinaryHead(nn.Module):
    """pooled fused features -> logits [B,2] (same layout as baseline's real/fake output)."""

    def __init__(self, in_dim: int):
        super().__init__()
        raise NotImplementedError("TODO(C2)")

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO(C2)")


class VocoderHead(nn.Module):
    """pooled fused features -> logits [B, num_vocoder_classes]; optional gradient reversal."""

    def __init__(self, in_dim: int, num_classes: int, gradient_reversal: bool = False):
        super().__init__()
        raise NotImplementedError("TODO(C2)")

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO(C2)")

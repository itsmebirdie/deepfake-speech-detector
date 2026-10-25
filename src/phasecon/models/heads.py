"""Classification heads. Owner: Member C (PR-11).

Both heads read ONE pooled vector per clip, [B, C_b], and have the same shape as the heads of the
released baseline (`Linear -> Linear`, no activation between them), so they can be initialised
from the baseline's pretrained heads (`load_pretrained`). Outputs are raw logits (no softmax).
"""
from typing import Dict, Optional

import torch
import torch.nn as nn


class GradReverse(torch.autograd.Function):
    """Gradient reversal layer (Ganin & Lempitsky 2015). Ablation (v).

    Forward is the identity. Backward multiplies the incoming gradient by `-lambd`, so whatever
    sits BEFORE this layer is trained to make the layers after it perform worse.
    """

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class _TwoLayerHead(nn.Module):
    """fc1 -> fc2 with no activation, the layout of the baseline's heads."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError(f"in_dim and out_dim must be positive, got {in_dim} and {out_dim}")
        hidden_dim = hidden_dim or in_dim
        self.in_dim, self.out_dim, self.hidden_dim = in_dim, out_dim, hidden_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def _check(self, h: torch.Tensor) -> None:
        if h.dim() != 2 or h.shape[1] != self.in_dim:
            raise ValueError(f"expected pooled features [B, {self.in_dim}], got {tuple(h.shape)}")

    def load_pretrained(self, state: Dict[str, torch.Tensor]) -> bool:
        """Copy fc1/fc2 weights (keys fc1.weight, fc1.bias, fc2.weight, fc2.bias), e.g. from
        `BaselineRawNet2.pretrained_heads()`. Returns False and changes nothing when any shape
        differs from this head's; raises KeyError when a key is missing."""
        own = {
            "fc1.weight": self.fc1.weight,
            "fc1.bias": self.fc1.bias,
            "fc2.weight": self.fc2.weight,
            "fc2.bias": self.fc2.bias,
        }
        missing = [k for k in own if k not in state]
        if missing:
            raise KeyError(f"pretrained head state is missing {missing}")
        if any(tuple(state[k].shape) != tuple(p.shape) for k, p in own.items()):
            return False
        with torch.no_grad():
            for k, p in own.items():
                p.copy_(state[k])
        return True

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        self._check(h)
        return self.fc2(self.fc1(h))


class BinaryHead(_TwoLayerHead):
    """pooled fused features [B, in_dim] -> logits [B, 2] in the repo order [bona fide, synthetic]
    (label 0 / 1), i.e. the baseline's two outputs swapped (see BaselineRawNet2.pretrained_heads)."""

    def __init__(self, in_dim: int, hidden_dim: Optional[int] = None):
        super().__init__(in_dim, 2, hidden_dim)


class VocoderHead(_TwoLayerHead):
    """pooled fused features [B, in_dim] -> logits [B, num_classes] (0 bona fide, 1..6 vocoders).

    With `gradient_reversal=True` (ablation v) the features pass through a gradient-reversal layer
    first: this head still learns to name the vocoder, but the layers below it are pushed to hide
    vocoder identity (scaled by `lambd`). The forward output is unchanged by the flag.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        gradient_reversal: bool = False,
        lambd: float = 1.0,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__(in_dim, num_classes, hidden_dim)
        self.gradient_reversal = bool(gradient_reversal)
        self.lambd = float(lambd)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        self._check(h)
        if self.gradient_reversal:
            h = grad_reverse(h, self.lambd)
        return self.fc2(self.fc1(h))

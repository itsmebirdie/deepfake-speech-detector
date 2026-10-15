"""Fusion blocks. Owner: Member C (step C3). Contract: docs/INTERFACES.md section 4."""
import torch
import torch.nn as nn


class GatedCrossAttentionFusion(nn.Module):
    """Eq. 5: H = H_b + sigmoid(W_g H_d) * Attn(H_b, H_d, H_d).
    forward(H_b [B,T_b,C_b], H_d [B,T_d,C_d]) -> H [B,T_b,C_b].
    Query = baseline stream. The gate must be computed on a T_b-aligned version of H_d.
    Invariants to test: gate -> 0 gives H == H_b; output shape == H_b shape; works if T_b != T_d."""

    def __init__(self, dim_b: int, dim_d: int, dim: int, num_heads: int, dropout: float):
        super().__init__()
        raise NotImplementedError("TODO(C3)")

    def forward(self, H_b: torch.Tensor, H_d: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO(C3)")


class ConcatFusion(nn.Module):
    """Ablation (iii): align time axes, concatenate along channels, project back to C_b."""

    def __init__(self, dim_b: int, dim_d: int):
        super().__init__()
        raise NotImplementedError("TODO(C3)")

    def forward(self, H_b: torch.Tensor, H_d: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO(C3)")

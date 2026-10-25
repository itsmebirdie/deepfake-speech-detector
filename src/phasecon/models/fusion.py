"""Fusion blocks. Owner: Member C (PR-12). Contract: docs/INTERFACES.md sections 3 and 4.

Both blocks take the baseline sequence H_b [B, T_b, C_b] and the phase sequence H_d [B, T_d, C_d]
and return a tensor with the shape of H_b, so the heads downstream do not care which block ran.
T_b and T_d may differ: attention handles it directly, everything else linearly resamples H_d
along time to T_b first (`align_time`).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def align_time(x: torch.Tensor, length: int) -> torch.Tensor:
    """Linearly resample [B, T, C] along time to [B, length, C] (no-op when T == length)."""
    if x.shape[1] == length:
        return x
    return F.interpolate(x.transpose(1, 2), size=length, mode="linear", align_corners=False).transpose(1, 2)


def _check_streams(H_b: torch.Tensor, H_d: torch.Tensor, dim_b: int, dim_d: int) -> None:
    if H_b.dim() != 3 or H_b.shape[2] != dim_b:
        raise ValueError(f"H_b must be [B, T_b, {dim_b}], got {tuple(H_b.shape)}")
    if H_d.dim() != 3 or H_d.shape[2] != dim_d:
        raise ValueError(f"H_d must be [B, T_d, {dim_d}], got {tuple(H_d.shape)}")
    if H_b.shape[0] != H_d.shape[0]:
        raise ValueError(f"batch sizes differ: H_b {H_b.shape[0]} vs H_d {H_d.shape[0]}")
    if H_b.shape[1] == 0 or H_d.shape[1] == 0:
        raise ValueError("H_b and H_d must have at least one time step")


class GatedCrossAttentionFusion(nn.Module):
    """Eq. 5: H = H_b + sigmoid(W_g H_d) * Attn(H_b, H_d, H_d).
    forward(H_b [B,T_b,C_b], H_d [B,T_d,C_d]) -> H [B,T_b,C_b].

    * Query = baseline stream, keys/values = phase stream, `dim` is the attention width.
    * The gate is computed on a copy of H_d resampled to T_b, so it is a per-step, per-channel
      value in (0, 1) of shape [B, T_b, C_b]. A zero gate gives H == H_b exactly.
    * Both inputs go through a LayerNorm before projection (pre-norm); H_b itself is added
      un-normalised, so the residual path is the baseline's own feature.
    * The gate bias starts at `gate_bias_init` (default -3, gate about 0.05), so training starts
      close to the baseline and the phase stream is let in only as far as it helps.
    """

    def __init__(
        self, dim_b: int, dim_d: int, dim: int, num_heads: int, dropout: float, gate_bias_init: float = -3.0
    ):
        super().__init__()
        if dim <= 0 or num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be a positive multiple of num_heads ({num_heads})")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.dim_b, self.dim_d, self.dim = dim_b, dim_d, dim
        self.norm_b = nn.LayerNorm(dim_b)
        self.norm_d = nn.LayerNorm(dim_d)
        self.q_proj = nn.Linear(dim_b, dim)
        self.k_proj = nn.Linear(dim_d, dim)
        self.v_proj = nn.Linear(dim_d, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(dim, dim_b)
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Linear(dim_d, dim_b)  # W_g
        nn.init.constant_(self.gate.bias, gate_bias_init)

    def compute_gate(self, H_d: torch.Tensor, t_b: int) -> torch.Tensor:
        """sigmoid(W_g H_d~) with H_d~ = H_d resampled to t_b steps: [B, t_b, C_b], in (0, 1)."""
        return torch.sigmoid(self.gate(align_time(self.norm_d(H_d), t_b)))

    def forward(self, H_b: torch.Tensor, H_d: torch.Tensor) -> torch.Tensor:
        _check_streams(H_b, H_d, self.dim_b, self.dim_d)
        d = self.norm_d(H_d)
        q = self.q_proj(self.norm_b(H_b))
        attn, _ = self.attn(q, self.k_proj(d), self.v_proj(d), need_weights=False)
        attn = self.drop(self.out_proj(attn))  # [B, T_b, C_b]
        gate = self.compute_gate(H_d, H_b.shape[1])
        return H_b + gate * attn


class ConcatFusion(nn.Module):
    """Ablation (iii): align time axes, concatenate along channels, project back to C_b.

    With `identity_init` (default) the projection starts as [I | small noise]: it passes H_b through
    and lets H_d in gradually, the same starting point as the gated block, so the ablation compares
    the fusion mechanism and not the initialisation. (The H_d columns are small but not zero so that
    gradient reaches the phase stream from the first step.)
    """

    def __init__(self, dim_b: int, dim_d: int, identity_init: bool = True):
        super().__init__()
        self.dim_b, self.dim_d = dim_b, dim_d
        self.proj = nn.Linear(dim_b + dim_d, dim_b)
        if identity_init:
            with torch.no_grad():
                self.proj.weight[:, :dim_b].copy_(torch.eye(dim_b))
                self.proj.weight[:, dim_b:].normal_(0.0, 0.02)
                self.proj.bias.zero_()

    def forward(self, H_b: torch.Tensor, H_d: torch.Tensor) -> torch.Tensor:
        _check_streams(H_b, H_d, self.dim_b, self.dim_d)
        return self.proj(torch.cat([H_b, align_time(H_d, H_b.shape[1])], dim=-1))

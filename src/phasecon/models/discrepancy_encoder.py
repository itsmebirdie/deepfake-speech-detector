"""Encoder from per-resolution discrepancy tensors to a temporal feature sequence. Owner: Member B (B6)."""
from typing import List

import torch
import torch.nn as nn


class DiscrepancyEncoder(nn.Module):
    """forward(D_list: list of [B,C_in,F_r,N_r]) -> H_d [B, T_d, out_dim].
    Per-resolution 2-D CNN -> pool over frequency -> interpolate time axis to a common T_d -> concat/project.
    T_d is fixed by config so the fusion block can align with the RawNet2 sequence length."""

    def __init__(self, in_channels: int, channels: List[int], out_dim: int, num_resolutions: int, t_out: int | None = None):
        super().__init__()
        raise NotImplementedError("TODO(B6)")

    def forward(self, D_list: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("TODO(B6)")

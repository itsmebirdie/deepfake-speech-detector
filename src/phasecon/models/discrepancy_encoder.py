"""Encoder from per-resolution discrepancy tensors to a temporal feature sequence. Owner: Member B (B6).

Input: one discrepancy tensor per STFT resolution (PR-16), D_r [B, C_in, F_r, N_r]
(C_in = 3 with the IF channel, 2 without). The resolutions have different F_r and N_r, so each gets its own branch:

    D_r --(2-D CNN, frequency halved after every block; time kept)--> [B, C_last, F', N_r]
        --(mean over frequency)--> [B, C_last, N_r]
        --(adaptive average pool over time)--> [B, C_last, T_d]

The R branch outputs are concatenated on channels ([B, R*C_last, T_d]) and a linear layer projects each
time step to out_dim, giving H_d [B, T_d, out_dim] (INTERFACES.md section 3).

T_d = model.phase_branch.encoder.t_out = 43: the RawNet2 sequence length T_b for 96,000 samples (PR-08), so the
fusion block sees two sequences of the same length. Average pooling (not picking frames) is used for the time
resampling because it is a downsampling here (N_r = 751 / 376 / 188 frames at 96k) and averaging avoids aliasing.
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_T_OUT = 43  # T_b for 96,000 samples at 24 kHz (docs/INTERFACES.md section 4)


def _branch(in_channels: int, channels: List[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    c_prev = in_channels
    for c in channels:
        layers += [
            nn.Conv2d(c_prev, c, kernel_size=3, padding=1),
            nn.GroupNorm(1, c),  # per-clip normalisation: no mixing between clips in a batch
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1), ceil_mode=True),  # halve frequency, keep every frame
        ]
        c_prev = c
    return nn.Sequential(*layers)


class DiscrepancyEncoder(nn.Module):
    """forward(D_list: list of [B,C_in,F_r,N_r]) -> H_d [B, T_d, out_dim].
    Per-resolution 2-D CNN -> pool over frequency -> interpolate time axis to a common T_d -> concat/project.
    T_d is fixed by config so the fusion block can align with the RawNet2 sequence length."""

    def __init__(self, in_channels: int, channels: List[int], out_dim: int, num_resolutions: int, t_out: int | None = None):
        super().__init__()
        if num_resolutions < 1 or not channels:
            raise ValueError("need at least one resolution and one CNN block")
        self.t_out = int(t_out or DEFAULT_T_OUT)
        self.branches = nn.ModuleList(_branch(in_channels, list(channels)) for _ in range(num_resolutions))
        self.proj = nn.Linear(num_resolutions * channels[-1], out_dim)

    def forward(self, D_list: List[torch.Tensor]) -> torch.Tensor:
        if len(D_list) != len(self.branches):
            raise ValueError(f"expected {len(self.branches)} resolutions, got {len(D_list)}")
        feats = []
        for D, branch in zip(D_list, self.branches):
            h = branch(D).mean(dim=2)                      # [B, C_last, N_r]
            feats.append(F.adaptive_avg_pool1d(h, self.t_out))  # [B, C_last, T_d]
        h = torch.cat(feats, dim=1).transpose(1, 2)        # [B, T_d, R*C_last]
        return self.proj(h)                                # [B, T_d, out_dim]

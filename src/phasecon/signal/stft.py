"""Differentiable multi-resolution STFT / iSTFT. Owner: Member B (step B1).

Everything runs in float32 (autocast is switched off inside), whatever precision the caller uses,
because phase is sensitive to rounding. Windows are periodic Hann windows of `win_length`;
`torch.istft` divides by the summed squared window, so S^-1(S(x)) == x up to float error whenever
the hop is at most half the window (true for every resolution in configs/model/phasecon.yaml).
"""
from dataclasses import dataclass
from typing import List, Sequence, Union

import torch
import torch.nn as nn


@dataclass(frozen=True)
class STFTConfig:
    n_fft: int
    hop: int
    win_length: int
    center: bool = True

    def __post_init__(self):
        if not (0 < self.hop <= self.win_length <= self.n_fft):
            raise ValueError(f"need 0 < hop <= win_length <= n_fft, got {self}")
        if self.hop * 2 > self.win_length:
            raise ValueError(f"hop {self.hop} > win_length/2: the inverse would not reconstruct the signal")


ResolutionSpec = Union[STFTConfig, Sequence[int]]


def _as_config(spec: ResolutionSpec) -> STFTConfig:
    if isinstance(spec, STFTConfig):
        return spec
    n_fft, hop, win_length = (int(v) for v in spec)  # (n_fft, hop, win_length) as in the YAML config
    return STFTConfig(n_fft=n_fft, hop=hop, win_length=win_length)


class MultiResSTFT(nn.Module):
    """forward(wav [B,T]) -> list of complex tensors, one per resolution, each [B, F_r, N_r]
    with F_r = n_fft//2 + 1. Always computed in float32 (disable autocast inside).
    inverse(X [B,F,N], r, length) -> wav [B,length]
    """

    def __init__(self, resolutions: Sequence[ResolutionSpec]):
        super().__init__()
        self.configs: List[STFTConfig] = [_as_config(r) for r in resolutions]
        if not self.configs:
            raise ValueError("MultiResSTFT needs at least one resolution")
        for i, c in enumerate(self.configs):
            # a buffer moves with .to(device) and is saved with the model, but is not trained
            self.register_buffer(f"window_{i}", torch.hann_window(c.win_length, periodic=True), persistent=False)

    def __len__(self) -> int:
        return len(self.configs)

    def window(self, r: int) -> torch.Tensor:
        return getattr(self, f"window_{r}")

    def stft(self, wav: torch.Tensor, r: int) -> torch.Tensor:
        """One resolution: wav [B,T] -> complex64 [B, n_fft//2+1, N]."""
        if wav.dim() != 2:
            raise ValueError(f"expected wav [B, T], got shape {tuple(wav.shape)}")
        c = self.configs[r]
        with torch.autocast(device_type=wav.device.type, enabled=False):
            return torch.stft(
                wav.float(), n_fft=c.n_fft, hop_length=c.hop, win_length=c.win_length,
                window=self.window(r).to(wav.device), center=c.center, pad_mode="reflect",
                normalized=False, onesided=True, return_complex=True,
            )

    def forward(self, wav: torch.Tensor) -> List[torch.Tensor]:
        return [self.stft(wav, r) for r in range(len(self.configs))]

    def inverse(self, X: torch.Tensor, r: int, length: int) -> torch.Tensor:
        """Complex STFT [B,F,N] at resolution r -> wav [B, length] (float32)."""
        c = self.configs[r]
        if X.dim() != 3 or X.shape[1] != c.n_fft // 2 + 1:
            raise ValueError(f"expected X [B, {c.n_fft // 2 + 1}, N] for resolution {r}, got {tuple(X.shape)}")
        with torch.autocast(device_type=X.device.type, enabled=False):
            return torch.istft(
                X.to(torch.complex64), n_fft=c.n_fft, hop_length=c.hop, win_length=c.win_length,
                window=self.window(r).to(X.device), center=c.center, normalized=False, onesided=True,
                length=length,
            )

    def project(self, X: torch.Tensor, r: int, length: int) -> torch.Tensor:
        """S(S^-1(X)): projection onto the set of consistent STFTs (Eq. 1)."""
        return self.stft(self.inverse(X, r, length), r)

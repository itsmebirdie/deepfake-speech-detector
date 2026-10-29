"""Unrolled Learnable Consistency Projection (Eqs. 1-3). Owner: Member B (step B4).

ULCP is fast Griffin-Lim (PR-05) unrolled for K steps, where each step may learn two things:
  * its own momentum alpha_k (starts at momentum_init, e.g. 0.99), and
  * a small convolutional correction g_theta_k on the [Re, Im] image of the STFT.

Each step k (C = current estimate, T = last magnitude-corrected projection):
    Z      = S(S^-1(C_{k-1}))                       projection onto consistent STFTs (Eq. 1)
    T_k    = A * Z / |Z|                            put the observed magnitude back (Eq. 3)
    C_k    = T_k + alpha_k (T_k - T_{k-1}) + g_theta_k(T_k)     momentum + learned correction (Eq. 2)
Output Y_K = T_K, so |Y_K| == A exactly, and phi_hat = angle(Y_K).

This is the order of fast Griffin-Lim (Perraudin et al. 2013; `signal/griffin_lim.py`): momentum is
applied AFTER the magnitude is re-imposed. With learnable=False (fixed alpha, no g_theta) the module
therefore reproduces `fast_griffin_lim` exactly, which is ablation (ii). The last conv of every
g_theta starts at zero, so a freshly built learnable ULCP also starts out equal to fast Griffin-Lim.

The output is T_K, which step K computes BEFORE its own momentum and correction. Those two would therefore
never affect the output and never receive a gradient, so the last step is built without them (fixed alpha,
no g_theta): every learnable parameter of ULCP is one that can actually learn (PR-27; found in PR-23).
"""
import torch
import torch.nn as nn

from phasecon.signal.residual import initial_phase
from phasecon.signal.stft import MultiResSTFT


class ULCPStep(nn.Module):
    """One unrolled iteration k: Z=S(S^-1(C)); T=A*Z/|Z|; C'=T+alpha_k(T-T_prev)+g_theta_k(T)."""

    def __init__(self, momentum_init: float, learnable_momentum: bool, correction_channels: int):
        super().__init__()
        alpha = torch.tensor(float(momentum_init))
        if learnable_momentum:
            self.alpha = nn.Parameter(alpha)
        else:
            self.register_buffer("alpha", alpha)
        self.correction = None
        if correction_channels > 0:
            last = nn.Conv2d(correction_channels, 2, kernel_size=3, padding=1)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            self.correction = nn.Sequential(
                nn.Conv2d(2, correction_channels, kernel_size=3, padding=1), nn.GELU(), last
            )

    def forward(self, C, T_prev, A, stft: MultiResSTFT, r: int, length: int):
        """Returns (C_next, T). g_theta acts on stacked [Re T, Im T] and returns a complex-compatible correction."""
        T = torch.polar(A, torch.angle(stft.project(C, r, length)))
        C_next = T + self.alpha * (T - T_prev)
        if self.correction is not None:
            ri = torch.stack([T.real, T.imag], dim=1)  # [B, 2, F, N]
            delta = self.correction(ri)
            C_next = C_next + torch.complex(delta[:, 0], delta[:, 1])
        return C_next, T


class ULCP(nn.Module):
    """forward(A [B,F,N] magnitude, stft, r, length) -> Y_K complex [B,F,N] with |Y_K| == A.
    phi_hat = angle(Y_K). Deterministic init (no random phase). K steps from config."""

    def __init__(self, num_steps: int, init: str, momentum_init: float,
                 learnable_momentum: bool, correction_channels: int, learnable: bool = True):
        super().__init__()
        if num_steps < 0:
            raise ValueError(f"num_steps must be >= 0, got {num_steps}")
        initial_phase((1, 2, 1), 2, init)  # validates the init name early
        self.init = init
        self.learnable = learnable
        # the last step's momentum/correction cannot reach the output T_K (see module docstring)
        self.steps = nn.ModuleList(
            ULCPStep(
                momentum_init,
                learnable_momentum and learnable and k < num_steps - 1,
                correction_channels if learnable and k < num_steps - 1 else 0,
            )
            for k in range(num_steps)
        )

    @classmethod
    def from_config(cls, ulcp_cfg: dict) -> "ULCP":
        """Build from `model.phase_branch.ulcp` in configs/model/phasecon.yaml."""
        return cls(num_steps=int(ulcp_cfg.get("num_steps", 5)), init=ulcp_cfg.get("init", "zeros"),
                   momentum_init=float(ulcp_cfg.get("momentum_init", 0.99)),
                   learnable_momentum=bool(ulcp_cfg.get("learnable_momentum", True)),
                   correction_channels=int(ulcp_cfg.get("correction_channels", 8)),
                   learnable=bool(ulcp_cfg.get("learnable", True)))

    def forward(self, A: torch.Tensor, stft: MultiResSTFT, r: int, length: int) -> torch.Tensor:
        A = A.float()
        phi0 = initial_phase(A.shape, stft.configs[r].n_fft, self.init, device=A.device)
        C = torch.polar(A, phi0)
        T = C
        for step in self.steps:
            C, T = step(C, T, A, stft, r, length)
        return T

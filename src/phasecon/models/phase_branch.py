"""Phase-discrepancy stream = STFT -> ULCP -> discrepancy tensors -> encoder. Owner: Member B (B7).
THIS IS THE B -> C HAND-OFF INTERFACE. Do not change the output keys without telling Member C.

For every STFT resolution r:
    X_r  = STFT_r(wav)                          complex [B, F_r, N_r]          (PR-03)
    Y_r  = ULCP_r(|X_r|)                        phase guessed from magnitude    (PR-14)
    D_r  = discrepancy(X_r, Y_r)                [B, 3, F_r, N_r]               (PR-16)
Then H_d = encoder([D_1 .. D_R])                [B, T_d, C_d]                  (PR-19)
and  cons_per_sample[i] = mean over r of mean over (f, t) of w * (1 - cos dphi)   (INTERFACES.md section 3).

Each resolution has its own ULCP (its own learnable momenta and corrections). The signal part (STFT, ULCP,
discrepancy) runs in fp32 with autocast off, so mixed-precision training cannot break the phase arithmetic.
Every operation acts on each clip separately (the encoder uses GroupNorm, not BatchNorm), so clip i's output
never depends on the other clips in the batch.

mode = raw_phase (ablation i): ULCP is skipped and the encoder sees the raw phase. There is no guess to compare
with, so cons_per_sample is zeros (and L_cons is then zero).
"""
from typing import Dict

import torch
import torch.nn as nn

from phasecon.models.discrepancy import build_discrepancy_tensor, confidence_weight, phase_discrepancy
from phasecon.models.discrepancy_encoder import DiscrepancyEncoder
from phasecon.models.ulcp import ULCP
from phasecon.signal.stft import MultiResSTFT

DEFAULT_RESOLUTIONS = [(512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)]


class PhaseBranch(nn.Module):
    """forward(wav [B,T]) -> dict:
        'H_d':            [B, T_d, C_d]   features for fusion
        'cons_per_sample':[B]             mean_{r,f,t} w * (1 - cos dphi)   (used by L_cons, masked to bona fide)
        'delta_phi':      list of [B,F_r,N_r]  (debug / visualisation only; may be omitted in training)
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        cfg = cfg or {}
        ulcp_cfg = cfg.get("ulcp", {}) or {}
        disc_cfg = cfg.get("discrepancy", {}) or {}
        enc_cfg = cfg.get("encoder", {}) or {}

        self.stft = MultiResSTFT([tuple(r) for r in cfg.get("stft_resolutions", DEFAULT_RESOLUTIONS)])
        self.num_resolutions = len(self.stft.configs)
        self.tau = float(disc_cfg.get("tau", 1e-3))
        self.use_if_channel = bool(disc_cfg.get("use_if_channel", True))
        self.mode = disc_cfg.get("mode", "residual")
        if self.mode not in ("residual", "raw_phase"):
            raise ValueError(f"model.phase_branch.discrepancy.mode must be residual or raw_phase, got {self.mode!r}")
        self.return_delta_phi = bool(cfg.get("return_delta_phi", False))

        self.ulcp = (
            nn.ModuleList(ULCP.from_config(ulcp_cfg) for _ in range(self.num_resolutions))
            if self.mode == "residual" else None
        )
        self.encoder = DiscrepancyEncoder(
            in_channels=3 if self.use_if_channel else 2,
            channels=list(enc_cfg.get("channels", [16, 32, 64])),
            out_dim=int(enc_cfg.get("out_dim", 128)),
            num_resolutions=self.num_resolutions,
            t_out=enc_cfg.get("t_out"),
        )

    def forward(self, wav: torch.Tensor) -> Dict[str, torch.Tensor]:
        if wav.dim() != 2:
            raise ValueError(f"PhaseBranch expects wav [B, T], got shape {tuple(wav.shape)}")
        length = wav.shape[-1]
        D_list, cons, dphis = [], [], []
        with torch.autocast(device_type=wav.device.type, enabled=False):
            wav = wav.float()
            for r, X in enumerate(self.stft(wav)):
                c = self.stft.configs[r]
                Y = self.ulcp[r](X.abs(), self.stft, r, length) if self.ulcp is not None else None
                D_list.append(build_discrepancy_tensor(
                    X, Y, tau=self.tau, hop=c.hop, n_fft=c.n_fft,
                    use_if_channel=self.use_if_channel, mode=self.mode,
                ))
                if Y is not None:
                    dphi = phase_discrepancy(X, Y)
                    w = confidence_weight(X.abs(), self.tau)
                    cons.append((w * (1 - torch.cos(dphi))).mean(dim=(1, 2)))
                    if self.return_delta_phi:
                        dphis.append(dphi)
        out = {"H_d": self.encoder(D_list)}
        out["cons_per_sample"] = (
            torch.stack(cons, dim=0).mean(dim=0) if cons else torch.zeros(wav.shape[0], device=wav.device)
        )
        if self.return_delta_phi and dphis:
            out["delta_phi"] = dphis
        return out

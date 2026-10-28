"""Phase-discrepancy tensor (Eq. 4). Owner: Member B (step B5).

For each STFT resolution the model gets an image-like tensor D_r with one channel per feature:
    w * cos(dphi), w * sin(dphi)    the phase residual as a unit vector (no 2*pi jumps), weighted by
                                    w = A / (A + tau) so near-silent bins (noise) contribute ~0
    w * d_psi                       how much faster/slower the observed phase advances than the
                                    estimated one (instantaneous-frequency difference), wrapped
PR-10 showed that w alone carries a magnitude pattern; keeping w as a multiplier (not a separate
channel) means the residual channels stay about phase, while the baseline already sees magnitude.
"""
import torch

from phasecon.signal.instantaneous_frequency import instantaneous_frequency, wrap_to_pi

MODES = ("residual", "raw_phase")


def confidence_weight(A: torch.Tensor, tau: float) -> torch.Tensor:
    """w = A / (A + tau), in [0,1). A [B,F,N] -> [B,F,N]."""
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    return A / (A + tau)


def phase_discrepancy(X: torch.Tensor, Y_K: torch.Tensor) -> torch.Tensor:
    """delta_phi = angle(X * conj(Y_K))  [B,F,N], in (-pi, pi].
    NOTE: the report's Eq. 4 writes X*Y^(K); the conjugate is required for a phase *difference*
    (the overline was likely lost in typesetting). Confirm with team and keep the conjugate."""
    return wrap_to_pi(torch.angle(X * torch.conj(Y_K)))


def build_discrepancy_tensor(
    X: torch.Tensor, Y_K: torch.Tensor, tau: float, hop: int, n_fft: int,
    use_if_channel: bool = True, mode: str = "residual",
) -> torch.Tensor:
    """D_r = [w cos dphi, w sin dphi, d_psi] -> [B,3,F,N] (or [B,2,F,N] if IF channel disabled).
    mode='raw_phase' (ablation i): replace dphi by observed phase angle(X), no ULCP involved.
    d_psi = IF(observed) - IF(estimated), also confidence-weighted; same wrap-safe treatment.

    d_psi is expressed as a phase advance per hop, hop * (IF_X - IF_Y) wrapped to (-pi, pi], so all
    channels share the same scale. In raw_phase mode it is the observed advance per hop, wrapped,
    and `Y_K` may be None.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    X = X.to(torch.complex64)
    w = confidence_weight(X.abs(), tau)
    phase_x = torch.angle(X)
    if mode == "residual":
        if Y_K is None:
            raise ValueError("mode='residual' needs Y_K (the ULCP estimate)")
        if Y_K.shape != X.shape:
            raise ValueError(f"Y_K {tuple(Y_K.shape)} does not match X {tuple(X.shape)}")
        Y_K = Y_K.to(torch.complex64)
        dphi = phase_discrepancy(X, Y_K)
    else:
        dphi = phase_x
    channels = [w * torch.cos(dphi), w * torch.sin(dphi)]
    if use_if_channel:
        if_x = instantaneous_frequency(phase_x, hop, n_fft)
        if mode == "residual":
            d_psi = wrap_to_pi(hop * (if_x - instantaneous_frequency(torch.angle(Y_K), hop, n_fft)))
        else:
            d_psi = wrap_to_pi(hop * if_x)
        channels.append(w * d_psi)
    return torch.stack(channels, dim=1)

"""Phase-discrepancy tensor (Eq. 4). Owner: Member B (step B5)."""
import torch


def confidence_weight(A: torch.Tensor, tau: float) -> torch.Tensor:
    """w = A / (A + tau), in [0,1). A [B,F,N] -> [B,F,N]."""
    raise NotImplementedError("TODO(B5)")


def phase_discrepancy(X: torch.Tensor, Y_K: torch.Tensor) -> torch.Tensor:
    """delta_phi = angle(X * conj(Y_K))  [B,F,N], in (-pi, pi].
    NOTE: the report's Eq. 4 writes X*Y^(K); the conjugate is required for a phase *difference*
    (the overline was likely lost in typesetting). Confirm with team and keep the conjugate."""
    raise NotImplementedError("TODO(B5)")


def build_discrepancy_tensor(
    X: torch.Tensor, Y_K: torch.Tensor, tau: float, hop: int, n_fft: int,
    use_if_channel: bool = True, mode: str = "residual",
) -> torch.Tensor:
    """D_r = [w cos dphi, w sin dphi, d_psi] -> [B,3,F,N] (or [B,2,F,N] if IF channel disabled).
    mode='raw_phase' (ablation i): replace dphi by observed phase angle(X), no ULCP involved.
    d_psi = IF(observed) - IF(estimated), also confidence-weighted; same wrap-safe treatment."""
    raise NotImplementedError("TODO(B5)")

"""Owner: Member B. Tests for models/discrepancy.py (PR-16)."""
import math

import pytest

torch = pytest.importorskip("torch")

from phasecon.models.discrepancy import (  # noqa: E402
    build_discrepancy_tensor,
    confidence_weight,
    phase_discrepancy,
)
from phasecon.signal.stft import MultiResSTFT  # noqa: E402

N_FFT, HOP = 512, 128


@pytest.fixture
def X():
    torch.manual_seed(0)
    stft = MultiResSTFT([(N_FFT, HOP, N_FFT)])
    return stft(torch.randn(2, 12000) * 0.1)[0]


def test_identical_phase_gives_zero_residual(X):
    """X == Y_K -> cos=1*w, sin=0, d_psi=0."""
    D = build_discrepancy_tensor(X, X.clone(), tau=1e-3, hop=HOP, n_fft=N_FFT)
    w = confidence_weight(X.abs(), 1e-3)
    assert torch.allclose(D[:, 0], w, atol=1e-6)
    assert torch.allclose(D[:, 1], torch.zeros_like(w), atol=1e-6)
    assert torch.allclose(D[:, 2], torch.zeros_like(w), atol=1e-6)
    Y = X * 3.7  # same phase, different magnitude: still zero residual
    assert torch.allclose(phase_discrepancy(X, Y), torch.zeros(1), atol=1e-6)


def test_wrap_invariance(X):
    """Adding 2*pi to the phase of Y_K leaves D unchanged."""
    torch.manual_seed(1)
    Y = torch.polar(X.abs(), torch.rand(X.shape) * 6 - 3)
    Y_shifted = torch.polar(Y.abs(), torch.angle(Y) + 2 * math.pi)
    D1 = build_discrepancy_tensor(X, Y, 1e-3, HOP, N_FFT)
    D2 = build_discrepancy_tensor(X, Y_shifted, 1e-3, HOP, N_FFT)
    assert torch.allclose(D1, D2, atol=1e-4)
    dphi = phase_discrepancy(X, Y)
    assert (dphi > -math.pi).all() and (dphi <= math.pi + 1e-6).all()


def test_confidence_weight_range():
    """0 <= w < 1 and monotone in A."""
    A = torch.logspace(-8, 4, 200)
    w = confidence_weight(A, 1e-3)
    assert (w >= 0).all() and (w < 1).all()
    assert (w[1:] >= w[:-1]).all()
    assert confidence_weight(torch.tensor(1e-3), 1e-3) == 0.5
    with pytest.raises(ValueError):
        confidence_weight(A, 0.0)


def test_low_energy_bins_suppressed(X):
    """Near-silent bins contribute ~0 to D and to cons_per_sample."""
    torch.manual_seed(2)
    quiet = X * 1e-8  # every bin far below tau (w < 1e-4)
    Y = torch.polar(quiet.abs(), torch.rand(X.shape) * 6 - 3)  # completely wrong phase
    D = build_discrepancy_tensor(quiet, Y, 1e-3, HOP, N_FFT)
    assert D.abs().max() < 1e-3
    cons = (confidence_weight(quiet.abs(), 1e-3) * (1 - torch.cos(phase_discrepancy(quiet, Y)))).mean()
    assert cons < 1e-3


def test_raw_phase_mode(X):
    """mode='raw_phase' bypasses ULCP and returns observed-phase channels."""
    D = build_discrepancy_tensor(X, None, 1e-3, HOP, N_FFT, mode="raw_phase")
    w = confidence_weight(X.abs(), 1e-3)
    assert torch.allclose(D[:, 0], w * torch.cos(torch.angle(X)), atol=1e-6)
    assert torch.allclose(D[:, 1], w * torch.sin(torch.angle(X)), atol=1e-6)
    with pytest.raises(ValueError):
        build_discrepancy_tensor(X, None, 1e-3, HOP, N_FFT, mode="residual")
    with pytest.raises(ValueError):
        build_discrepancy_tensor(X, X, 1e-3, HOP, N_FFT, mode="magnitude")


def test_output_shape(X):
    """[B,3,F,N] (or [B,2,F,N] without IF channel)."""
    B, F, N = X.shape
    assert build_discrepancy_tensor(X, X, 1e-3, HOP, N_FFT).shape == (B, 3, F, N)
    assert build_discrepancy_tensor(X, X, 1e-3, HOP, N_FFT, use_if_channel=False).shape == (B, 2, F, N)
    D = build_discrepancy_tensor(X, X, 1e-3, HOP, N_FFT)
    assert D.dtype == torch.float32 and torch.isfinite(D).all()

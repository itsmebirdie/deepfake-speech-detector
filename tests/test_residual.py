"""Owner: Member B. Tests for signal/residual.py (PR-10 feasibility helper)."""
import math

import pytest

torch = pytest.importorskip("torch")

from phasecon.signal.residual import clip_score, initial_phase, phase_residual_map  # noqa: E402
from phasecon.signal.stft import MultiResSTFT  # noqa: E402


def test_residual_map_shape_range_and_determinism():
    stft = MultiResSTFT([(512, 128, 512), (1024, 256, 1024)])
    torch.manual_seed(0)
    wav = torch.randn(2, 12000) * 0.1
    for r, (n_fft, hop) in enumerate([(512, 128), (1024, 256)]):
        for init in ("zeros", "linear_group_delay"):
            m = phase_residual_map(wav, stft, r, n_iter=4, init=init)
            assert m.shape == (2, n_fft // 2 + 1, 12000 // hop + 1)
            assert (m >= 0).all() and (m <= 2).all()
            assert torch.equal(m, phase_residual_map(wav, stft, r, n_iter=4, init=init))  # deterministic
            assert clip_score(m).shape == (2,)


def test_residual_is_zero_when_rebuilt_phase_is_the_true_phase(monkeypatch):
    """If Griffin-Lim returned the clip's own STFT, every bin would have dphi = 0 -> map = 0."""
    import phasecon.signal.residual as residual

    stft = MultiResSTFT([(512, 128, 512)])
    wav = torch.randn(1, 6000) * 0.1
    monkeypatch.setattr(residual, "fast_griffin_lim", lambda A, s, r, length, **kw: s.stft(wav, r))
    assert torch.allclose(phase_residual_map(wav, stft, 0), torch.zeros(1), atol=1e-5)


def test_initial_phase():
    z = initial_phase((1, 5, 3), 8, "zeros")
    assert torch.equal(z, torch.zeros(1, 5, 3))
    g = initial_phase((1, 5, 3), 8, "linear_group_delay")
    assert torch.allclose(g[0, :, 0], -math.pi * torch.arange(5.0))
    assert torch.equal(g[..., 0], g[..., 2])  # same in every frame
    with pytest.raises(ValueError):
        initial_phase((1, 5, 3), 8, "random")

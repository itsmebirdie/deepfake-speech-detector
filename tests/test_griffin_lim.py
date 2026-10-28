"""Owner: Member B. Tests for signal/griffin_lim.py and signal/instantaneous_frequency.py (PR-05)."""
import math

import pytest

torch = pytest.importorskip("torch")

from phasecon.signal.griffin_lim import fast_griffin_lim, spectral_convergence  # noqa: E402
from phasecon.signal.instantaneous_frequency import instantaneous_frequency, wrap_to_pi  # noqa: E402
from phasecon.signal.stft import MultiResSTFT  # noqa: E402

SR = 24000
T = SR  # 1 s


def speech_like(seed=0):
    """A voiced-speech stand-in (no dataset needed): a gliding pitch with harmonics, an envelope, a little noise."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(T) / SR
    f0 = 140 + 40 * torch.sin(2 * math.pi * 1.5 * t)
    phase = 2 * math.pi * torch.cumsum(f0, 0) / SR
    x = sum(torch.sin(h * phase) / h for h in range(1, 12))
    env = 0.5 + 0.5 * torch.sin(2 * math.pi * 3 * t).clamp(min=0)
    return (x * env + 0.01 * torch.randn(T, generator=g)).unsqueeze(0) * 0.1


def test_gl_preserves_magnitude():
    """Returned STFT has |Y| == input magnitude."""
    stft = MultiResSTFT([(512, 128, 512)])
    mag = stft(speech_like())[0].abs()
    for n_iter, momentum in [(0, 0.99), (5, 0.0), (8, 0.99)]:
        Y = fast_griffin_lim(mag, stft, 0, T, n_iter=n_iter, momentum=momentum)
        assert Y.shape == mag.shape and Y.dtype == torch.complex64
        assert torch.allclose(Y.abs(), mag, atol=1e-5 * mag.max().item())


def test_gl_spectral_convergence_improves():
    """Spectral convergence decreases with more iterations on a real clip.

    (A synthetic speech-like clip, so CI does not need the dataset; PR-10 runs it on LibriSeVoc.)"""
    stft = MultiResSTFT([(512, 128, 512), (1024, 256, 1024)])
    for r in range(2):
        mag = stft(speech_like(r))[r].abs()
        sc = [spectral_convergence(fast_griffin_lim(mag, stft, r, T, n_iter=n), mag, stft, r, T).item()
              for n in (0, 4, 16, 48)]
        assert sc[0] > sc[1] > sc[2] > sc[3]
        assert sc[3] < 0.5 * sc[0]
        fast = spectral_convergence(fast_griffin_lim(mag, stft, r, T, n_iter=16), mag, stft, r, T)
        plain = spectral_convergence(fast_griffin_lim(mag, stft, r, T, n_iter=16, momentum=0.0), mag, stft, r, T)
        assert fast < plain  # momentum helps


def test_instantaneous_frequency_of_sinusoid():
    """IF of a pure tone is constant and equals the tone frequency."""
    for f0, (n_fft, hop) in [(1000.0, (512, 128)), (3137.5, (1024, 256)), (440.0, (2048, 512))]:
        stft = MultiResSTFT([(n_fft, hop, n_fft)])
        t = torch.arange(T) / SR
        X = stft(torch.sin(2 * math.pi * f0 * t).unsqueeze(0))[0]
        inst = instantaneous_frequency(torch.angle(X), hop, n_fft)
        assert inst.shape == X.shape and torch.all(inst[..., 0] == 0)
        k = int(X.abs()[0, :, 10].argmax())  # the bin holding the tone
        hz = inst[0, k, 4:-4] * SR / (2 * math.pi)  # skip the padded edge frames
        assert torch.allclose(hz, torch.full_like(hz, f0), atol=0.5)


def test_wrap_to_pi_range():
    """Output always in (-pi, pi]."""
    x = torch.linspace(-50, 50, 100_001, dtype=torch.float64)
    x = torch.cat([x, torch.tensor([math.pi, -math.pi, 3 * math.pi, -3 * math.pi, 0.0], dtype=torch.float64)])
    y = wrap_to_pi(x)
    assert (y > -math.pi).all() and (y <= math.pi).all()
    assert torch.allclose(torch.cos(y), torch.cos(x), atol=1e-9)  # same angle
    assert torch.allclose(torch.sin(y), torch.sin(x), atol=1e-9)
    assert wrap_to_pi(torch.tensor(-math.pi, dtype=torch.float64)) == math.pi


def test_gl_rejects_bad_input():
    stft = MultiResSTFT([(512, 128, 512)])
    mag = torch.ones(1, 257, 10)
    with pytest.raises(ValueError):
        fast_griffin_lim(mag, stft, 0, 1152, n_iter=-1)
    with pytest.raises(ValueError):
        fast_griffin_lim(mag, stft, 0, 1152, init_phase=torch.zeros(1, 257, 9))
    with pytest.raises(ValueError):
        instantaneous_frequency(torch.zeros(1, 100, 5), 128, 512)

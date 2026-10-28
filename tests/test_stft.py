"""Owner: Member B. Tests for signal/stft.py (PR-03). Need only PyTorch, no dataset."""
import pytest

torch = pytest.importorskip("torch")

from phasecon.signal.stft import MultiResSTFT, STFTConfig  # noqa: E402

# the resolutions in configs/model/phasecon.yaml: (n_fft, hop, win_length)
RESOLUTIONS = [(512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)]
T = 24000  # 1 s at 24 kHz keeps the tests fast; the real input is 96,000 samples


@pytest.fixture
def stft():
    return MultiResSTFT(RESOLUTIONS)


@pytest.fixture
def wav():
    torch.manual_seed(0)
    return torch.randn(2, T) * 0.1


def test_istft_roundtrip(stft, wav):
    """S^-1(S(x)) reconstructs x for every configured resolution."""
    for r, X in enumerate(stft(wav)):
        back = stft.inverse(X, r, T)
        assert back.shape == wav.shape
        assert torch.allclose(back, wav, atol=1e-5)


def test_project_is_idempotent(stft):
    """S(S^-1(S(S^-1(X)))) == S(S^-1(X))."""
    torch.manual_seed(1)
    for r, c in enumerate(stft.configs):
        n_frames = T // c.hop + 1
        X = torch.randn(2, c.n_fft // 2 + 1, n_frames, dtype=torch.complex64)  # random: NOT a consistent STFT
        once = stft.project(X, r, T)
        twice = stft.project(once, r, T)
        assert not torch.allclose(once, X, atol=1e-3)  # projecting a random spectrogram changes it
        assert torch.allclose(twice, once, atol=1e-4 * once.abs().max().item())


def test_output_shapes(stft, wav):
    """Complex [B, n_fft//2+1, N] per resolution."""
    out = stft(wav)
    assert len(out) == len(RESOLUTIONS)
    for X, (n_fft, hop, _) in zip(out, RESOLUTIONS):
        assert X.shape == (2, n_fft // 2 + 1, T // hop + 1)
        assert X.dtype == torch.complex64


def test_gradients_flow(stft, wav):
    """Gradient reaches the input waveform."""
    wav = wav.clone().requires_grad_(True)
    loss = sum(X.abs().pow(2).mean() + X.angle().cos().mean() for X in stft(wav))
    loss = loss + stft.project(stft(wav)[0], 0, T).abs().mean()
    loss.backward()
    assert wav.grad is not None
    assert torch.isfinite(wav.grad).all() and wav.grad.abs().sum() > 0


def test_fp32_under_autocast(stft, wav):
    """Output dtype is complex64 even inside torch.autocast."""
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = stft(wav.to(torch.bfloat16))
        back = stft.inverse(out[0], 0, T)
    assert all(X.dtype == torch.complex64 for X in out)
    assert back.dtype == torch.float32


def test_bad_configs_are_rejected():
    with pytest.raises(ValueError):
        STFTConfig(n_fft=512, hop=300, win_length=512)  # hop > win/2: no perfect reconstruction
    with pytest.raises(ValueError):
        MultiResSTFT([])
    with pytest.raises(ValueError):
        MultiResSTFT(RESOLUTIONS).stft(torch.zeros(T), 0)  # needs [B, T]

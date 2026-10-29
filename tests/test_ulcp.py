"""Owner: Member B. Tests for models/ulcp.py (PR-14)."""
import pytest

torch = pytest.importorskip("torch")

from phasecon.models.ulcp import ULCP  # noqa: E402
from phasecon.signal.griffin_lim import fast_griffin_lim  # noqa: E402
from phasecon.signal.residual import initial_phase  # noqa: E402
from phasecon.signal.stft import MultiResSTFT  # noqa: E402

T = 6000


@pytest.fixture
def stft():
    return MultiResSTFT([(512, 128, 512), (1024, 256, 1024)])


@pytest.fixture
def A(stft):
    torch.manual_seed(0)
    return stft(torch.randn(2, T) * 0.1)[0].abs()


def make(K, learnable=True, init="zeros", channels=8):
    return ULCP(num_steps=K, init=init, momentum_init=0.99, learnable_momentum=True,
                correction_channels=channels, learnable=learnable)


def test_k0_returns_initialisation(stft, A):
    """With K=0 the output equals the deterministic init."""
    for init in ("zeros", "linear_group_delay"):
        Y = make(0, init=init)(A, stft, 0, T)
        assert torch.allclose(Y, torch.polar(A, initial_phase(A.shape, 512, init)))


def test_magnitude_is_preserved(stft, A):
    """|Y_K| == A for every K (Eq. 3)."""
    torch.manual_seed(1)
    for K in (1, 3, 5):
        ulcp = make(K)
        for step in ulcp.steps[:-1]:  # perturb the learnable parts so the test is not trivially the GL path
            step.alpha.data.fill_(0.7)
            torch.nn.init.normal_(step.correction[-1].weight, std=1e-2)
        Y = ulcp(A, stft, 0, T)
        assert torch.allclose(Y.abs(), A, atol=1e-5 * A.max().item())


def test_matches_fixed_fgl_when_not_learnable(stft, A):
    """alpha fixed, g_theta disabled -> equals fast Griffin-Lim reference."""
    for r, Ar in enumerate([A, stft(torch.randn(2, T) * 0.1)[1].abs()]):
        for K in (1, 4, 8):
            fixed = make(K, learnable=False)(Ar, stft, r, T)
            reference = fast_griffin_lim(Ar, stft, r, T, n_iter=K, momentum=0.99)
            assert torch.allclose(fixed, reference, atol=1e-5)
            assert torch.allclose(make(K, learnable=True)(Ar, stft, r, T), reference, atol=1e-5)  # zero-init g_theta
    assert sum(p.numel() for p in make(5, learnable=False).parameters()) == 0


def test_deterministic(stft, A):
    """Same input twice -> identical output (no random init)."""
    torch.manual_seed(2)
    ulcp = make(5)
    assert torch.equal(ulcp(A, stft, 0, T), ulcp(A, stft, 0, T))


def test_gradcheck_small(stft):
    """Gradients flow to alpha_k and g_theta on a tiny input."""
    torch.manual_seed(3)
    wav = torch.randn(1, 2048) * 0.1
    A = stft(wav)[0].abs()
    ulcp = make(3)
    X = stft(wav)[0]
    loss = (1 - torch.cos(torch.angle(X * torch.conj(ulcp(A, stft, 0, 2048))))).mean()
    loss.backward()
    for step in ulcp.steps[:-1]:
        assert step.alpha.grad is not None and step.alpha.grad.abs() > 0
        assert step.correction[-1].weight.grad.abs().sum() > 0
    # no dead parameters: the last step (whose momentum/correction cannot reach T_K) has none
    assert ulcp.steps[-1].correction is None and not isinstance(ulcp.steps[-1].alpha, torch.nn.Parameter)
    assert all(p.grad is not None for p in ulcp.parameters())


def saved_bytes(ulcp, A, stft):
    """Bytes autograd keeps for the backward pass: the memory that grows with K."""
    total = 0

    def pack(t):
        nonlocal total
        total += t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        ulcp(A.requires_grad_(True), stft, 0, T).abs().sum()
    return total


def test_memory_scales_with_k(stft, A):
    """Peak memory grows ~linearly in K (profile guard)."""
    m = {K: saved_bytes(make(K), A.clone(), stft) for K in (3, 5, 8)}
    per_step = (m[8] - m[3]) / 5
    assert per_step > 0
    assert abs((m[5] - m[3]) / 2 - per_step) <= 0.1 * per_step  # equal cost per extra step

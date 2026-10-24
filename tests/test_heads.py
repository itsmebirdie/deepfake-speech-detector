"""Owner: Member C (PR-11). Tests for models/heads.py (no baseline repo needed)."""
import pytest

torch = pytest.importorskip("torch")

from phasecon.models.heads import BinaryHead, GradReverse, VocoderHead, grad_reverse  # noqa: E402


def test_head_output_shapes():
    h = torch.randn(5, 32)
    assert BinaryHead(32)(h).shape == (5, 2)
    assert VocoderHead(32, 7)(h).shape == (5, 7)
    assert VocoderHead(32, 7, hidden_dim=16)(h).shape == (5, 7)
    with pytest.raises(ValueError):
        BinaryHead(32)(torch.randn(5, 31))  # wrong feature size
    with pytest.raises(ValueError):
        BinaryHead(32)(torch.randn(5, 4, 32))  # not pooled
    with pytest.raises(ValueError):
        BinaryHead(0)


def test_gradient_reversal_flips_the_sign():
    x = torch.randn(4, 6, requires_grad=True)
    w = torch.randn(4, 6)
    (grad_reverse(x, 1.0) * w).sum().backward()
    assert torch.allclose(x.grad, -w)  # d/dx of <x, w> is w, reversed
    x.grad = None
    (GradReverse.apply(x, 0.25) * w).sum().backward()
    assert torch.allclose(x.grad, -0.25 * w)  # scaled by lambda
    assert torch.equal(grad_reverse(x, 3.0), x)  # identity in the forward pass


def test_vocoder_head_gradient_reversal_flag():
    plain = VocoderHead(8, 7, gradient_reversal=False)
    rev = VocoderHead(8, 7, gradient_reversal=True, lambd=1.0)
    rev.load_state_dict(plain.state_dict())
    h = torch.randn(3, 8)
    assert torch.equal(plain(h), rev(h))  # forward is unaffected
    target = torch.tensor([0, 3, 6])
    h1 = h.clone().requires_grad_(True)
    h2 = h.clone().requires_grad_(True)
    torch.nn.functional.cross_entropy(plain(h1), target).backward()
    torch.nn.functional.cross_entropy(rev(h2), target).backward()
    assert torch.allclose(h2.grad, -h1.grad, atol=1e-6)  # features get the opposite gradient
    for a, b in zip(plain.parameters(), rev.parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-6)  # the head itself still learns normally


def test_load_pretrained():
    head = BinaryHead(8)
    other = BinaryHead(8)
    assert head.load_pretrained({k: v.clone() for k, v in other.state_dict().items()})
    assert torch.equal(head(torch.ones(2, 8)), other(torch.ones(2, 8)))
    before = {k: v.clone() for k, v in head.state_dict().items()}
    assert not head.load_pretrained(BinaryHead(9).state_dict())  # shape mismatch: untouched
    assert all(torch.equal(before[k], v) for k, v in head.state_dict().items())
    with pytest.raises(KeyError):
        head.load_pretrained({"fc1.weight": torch.zeros(8, 8)})
    # a head with a different output size cannot take binary weights
    assert not VocoderHead(8, 7).load_pretrained(other.state_dict())


def test_heads_train():
    """A head can fit a separable toy problem (guards against dead gradients)."""
    torch.manual_seed(0)
    head = BinaryHead(4)
    x = torch.randn(64, 4)
    y = (x[:, 0] > 0).long()
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    for _ in range(100):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.1

"""Owner: Member C (PR-15). Tests for losses/classification.py and losses/combined.py.

The cons term uses Member B's real `consistency_loss` (PR-22); its own tests are in test_phase_branch.py.
"""
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from phasecon.losses.classification import binary_loss, vocoder_loss  # noqa: E402
from phasecon.losses.combined import total_loss  # noqa: E402


def make_batch(b=6):
    torch.manual_seed(0)
    vid = torch.tensor([0, 1, 2, 3, 0, 6])[:b]
    return {"label": (vid > 0).long(), "vocoder_id": vid}


def make_out(b=6, cons=True, voc=True):
    torch.manual_seed(1)
    out = {"logits_bin": torch.randn(b, 2, requires_grad=True)}
    if voc:
        out["logits_voc"] = torch.randn(b, 7, requires_grad=True)
    if cons:
        out["cons_per_sample"] = torch.rand(b, requires_grad=True)
    return out


def test_binary_and_vocoder_loss_values():
    logits = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    label = torch.tensor([0, 1])
    expected = -(torch.log_softmax(logits, 1)[[0, 1], [0, 1]]).mean()
    assert binary_loss(logits, label).item() == pytest.approx(expected.item())
    voc = torch.randn(4, 7)
    vid = torch.tensor([0, 3, 6, 2])
    assert vocoder_loss(voc, vid).item() == pytest.approx(F.cross_entropy(voc, vid).item())
    # log-probabilities (the baseline wrapper's output) give the same loss as raw logits
    assert binary_loss(torch.log_softmax(logits, 1), label).item() == pytest.approx(binary_loss(logits, label).item())
    # float32 even for half-precision logits
    assert binary_loss(logits.bfloat16(), label).dtype == torch.float32


def test_class_weights_and_bad_shapes():
    logits, label = torch.randn(5, 2), torch.tensor([0, 1, 1, 0, 1])
    w = torch.tensor([1.0, 6.0])
    assert binary_loss(logits, label, w).item() == pytest.approx(F.cross_entropy(logits, label, weight=w).item())
    assert binary_loss(logits, label, w).item() != pytest.approx(binary_loss(logits, label).item())
    with pytest.raises(ValueError):
        binary_loss(torch.randn(5, 3), label)
    with pytest.raises(ValueError):
        binary_loss(logits, label[:4])
    with pytest.raises(ValueError):
        vocoder_loss(torch.randn(5), label)


def test_total_loss_composition():
    """total == bin + beta*voc + mu*cons on synthetic inputs."""
    out, batch = make_out(), make_batch()
    beta, mu = 0.5, 0.1
    loss, parts = total_loss(out, batch, beta=beta, mu=mu)
    l_bin = F.cross_entropy(out["logits_bin"], batch["label"])
    l_voc = F.cross_entropy(out["logits_voc"], batch["vocoder_id"])
    l_cons = out["cons_per_sample"][batch["label"] == 0].mean()
    assert loss.item() == pytest.approx((l_bin + beta * l_voc + mu * l_cons).item(), rel=1e-6)
    assert set(parts) == {"bin", "voc", "cons"}
    assert parts["bin"] == pytest.approx(l_bin.item()) and parts["voc"] == pytest.approx(l_voc.item())
    assert parts["cons"] == pytest.approx(l_cons.item())
    assert all(isinstance(v, float) for v in parts.values())
    loss.backward()
    assert all(t.grad is not None for t in out.values())


def test_missing_terms_are_skipped():
    """Baseline-only outputs (no cons key) do not crash."""
    batch = make_batch()
    loss, parts = total_loss(make_out(cons=False), batch, beta=0.5, mu=0.1)
    assert set(parts) == {"bin", "voc"}
    loss, parts = total_loss(make_out(cons=False, voc=False), batch, beta=0.5, mu=0.1)
    assert set(parts) == {"bin"}
    assert loss.item() == pytest.approx(parts["bin"])
    with pytest.raises(KeyError):
        total_loss({"logits_voc": torch.zeros(6, 7)}, batch, 0.5, 0.1)
    with pytest.raises(KeyError):
        total_loss(make_out(cons=False), {"label": batch["label"]}, 0.5, 0.1)
    with pytest.raises(ValueError):
        total_loss(make_out(), batch, beta=-1.0, mu=0.1)


def test_mu_zero_removes_cons_gradient():
    """mu=0 -> no gradient through cons term."""
    out, batch = make_out(), make_batch()
    loss, parts = total_loss(out, batch, beta=0.5, mu=0.0)
    assert "cons" not in parts
    loss.backward()
    assert out["cons_per_sample"].grad is None
    assert out["logits_bin"].grad is not None


def test_mu_zero_skips_the_cons_term():
    """mu=0 leaves the cons term out entirely (baseline-only and ablation iv runs)."""
    out, batch = make_out(), make_batch()
    loss, parts = total_loss(out, batch, beta=0.5, mu=0.0)
    assert torch.isfinite(loss) and "cons" not in parts


def test_beta_zero_removes_vocoder_gradient():
    out, batch = make_out(cons=False), make_batch()
    total_loss(out, batch, beta=0.0, mu=0.0)[0].backward()
    assert out["logits_voc"].grad is None


def test_cons_term_gradient_reaches_bona_fide_only():
    out, batch = make_out(), make_batch()
    total_loss(out, batch, beta=0.0, mu=1.0)[0].backward()
    g = out["cons_per_sample"].grad
    assert (g[batch["label"] == 0] != 0).all()
    assert (g[batch["label"] == 1] == 0).all()

"""Owner: Member C (PR-12). Tests for models/fusion.py (random H_d of the documented shape)."""
import pytest

torch = pytest.importorskip("torch")

from phasecon.models.fusion import ConcatFusion, GatedCrossAttentionFusion, align_time  # noqa: E402

C_B, C_D = 32, 12


def make_gated(**kw):
    args = dict(dim_b=C_B, dim_d=C_D, dim=16, num_heads=4, dropout=0.1)
    args.update(kw)
    return GatedCrossAttentionFusion(**args)


@pytest.mark.parametrize("t_b,t_d", [(7, 5), (7, 7), (5, 40), (1, 9), (9, 1)])
def test_output_shape_matches_hb(t_b, t_d):
    """H.shape == H_b.shape for T_b != T_d (and for equal lengths and length-1 sequences)."""
    H_b, H_d = torch.randn(3, t_b, C_B), torch.randn(3, t_d, C_D)
    assert make_gated()(H_b, H_d).shape == H_b.shape
    assert ConcatFusion(C_B, C_D)(H_b, H_d).shape == H_b.shape


def test_zero_gate_is_identity():
    """Force gate to 0 -> H == H_b."""
    fusion = make_gated().eval()
    with torch.no_grad():
        fusion.gate.weight.zero_()
        fusion.gate.bias.fill_(-1e4)  # sigmoid(-1e4) == 0 exactly in float32
    H_b, H_d = torch.randn(2, 6, C_B), torch.randn(2, 9, C_D)
    assert torch.equal(fusion(H_b, H_d), H_b)
    # and a non-zero gate really changes the output (the block is not an identity by accident)
    with torch.no_grad():
        fusion.gate.bias.fill_(0.0)
    assert not torch.allclose(fusion(H_b, H_d), H_b)


def test_default_start_is_close_to_the_baseline():
    """gate_bias_init keeps the fused features near H_b at the start of training."""
    torch.manual_seed(0)
    fusion = make_gated().eval()
    H_b, H_d = torch.randn(4, 8, C_B), torch.randn(4, 8, C_D)
    rel = (fusion(H_b, H_d) - H_b).norm() / H_b.norm()
    assert rel < 0.2
    open_gate = make_gated(gate_bias_init=0.0).eval()
    assert (open_gate(H_b, H_d) - H_b).norm() / H_b.norm() > rel


def test_gate_in_unit_interval():
    """sigmoid gate in (0,1), shaped [B, T_b, C_b]."""
    fusion = make_gated()
    gate = fusion.compute_gate(torch.randn(3, 9, C_D) * 10, 6)
    assert gate.shape == (3, 6, C_B)
    assert (gate > 0).all() and (gate < 1).all()
    # the default bias makes the gate small at initialisation
    assert gate.mean() < 0.2


def test_concat_fusion_shapes():
    """ConcatFusion output matches H_b shape, and starts as (almost) the identity on H_b."""
    fusion = ConcatFusion(C_B, C_D)
    H_b, H_d = torch.randn(2, 6, C_B), torch.randn(2, 11, C_D)
    out = fusion(H_b, H_d)
    assert out.shape == H_b.shape
    assert (out - H_b).norm() / H_b.norm() < 0.2
    plain = ConcatFusion(C_B, C_D, identity_init=False)
    assert plain(H_b, H_d).shape == H_b.shape
    assert not torch.allclose(plain(H_b, H_d), H_b, atol=0.5)  # random init is not near identity


@pytest.mark.parametrize("make", [make_gated, lambda: ConcatFusion(C_B, C_D)])
def test_gradients_reach_both_streams(make):
    """Gradients reach H_b and H_d (non-zero) and every parameter of the block."""
    fusion = make()
    H_b = torch.randn(2, 6, C_B, requires_grad=True)
    H_d = torch.randn(2, 9, C_D, requires_grad=True)
    fusion(H_b, H_d).pow(2).sum().backward()
    assert H_b.grad is not None and H_b.grad.abs().sum() > 0
    assert H_d.grad is not None and H_d.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fusion.parameters())


def test_align_time():
    x = torch.arange(10, dtype=torch.float32).view(1, 10, 1)
    assert align_time(x, 10) is x
    y = align_time(x, 5)
    assert y.shape == (1, 5, 1) and (y.diff(dim=1) > 0).all()  # still increasing
    assert align_time(torch.ones(2, 1, 3), 4).shape == (2, 4, 3)
    assert torch.allclose(align_time(torch.ones(2, 1, 3), 4), torch.ones(2, 4, 3))


def test_bad_arguments():
    with pytest.raises(ValueError):
        make_gated(dim=15)  # not divisible by num_heads
    with pytest.raises(ValueError):
        make_gated(dropout=1.0)
    fusion = make_gated()
    with pytest.raises(ValueError):
        fusion(torch.randn(2, 6, C_B + 1), torch.randn(2, 6, C_D))
    with pytest.raises(ValueError):
        fusion(torch.randn(2, 6, C_B), torch.randn(3, 6, C_D))  # batch mismatch
    with pytest.raises(ValueError):
        ConcatFusion(C_B, C_D)(torch.randn(2, 6, C_B), torch.randn(2, 0, C_D))


def test_eval_is_deterministic_and_batch_independent():
    fusion = make_gated().eval()
    H_b, H_d = torch.randn(4, 6, C_B), torch.randn(4, 9, C_D)
    with torch.no_grad():
        full = fusion(H_b, H_d)
        assert torch.equal(full, fusion(H_b, H_d))
        assert torch.allclose(full[1:2], fusion(H_b[1:2], H_d[1:2]), atol=1e-6)


def test_runs_under_autocast():
    fusion = make_gated().eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = fusion(torch.randn(2, 6, C_B), torch.randn(2, 9, C_D))
    assert out.shape == (2, 6, C_B) and torch.isfinite(out.float()).all()

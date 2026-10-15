"""Owner: Member B. Tests are specified but skipped until the module is implemented."""
import pytest


def test_k0_returns_initialisation():
    """With K=0 the output equals the deterministic init."""
    pytest.skip("not implemented yet")


def test_magnitude_is_preserved():
    """|Y_K| == A for every K (Eq. 3)."""
    pytest.skip("not implemented yet")


def test_matches_fixed_fgl_when_not_learnable():
    """alpha fixed, g_theta disabled -> equals fast Griffin-Lim reference."""
    pytest.skip("not implemented yet")


def test_deterministic():
    """Same input twice -> identical output (no random init)."""
    pytest.skip("not implemented yet")


def test_gradcheck_small():
    """Gradients flow to alpha_k and g_theta on a tiny input."""
    pytest.skip("not implemented yet")


def test_memory_scales_with_k():
    """Peak memory grows ~linearly in K (profile guard)."""
    pytest.skip("not implemented yet")

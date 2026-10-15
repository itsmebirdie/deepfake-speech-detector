"""Owner: Member C. Tests are specified but skipped until the module is implemented."""
import pytest


def test_output_shape_matches_hb():
    """H.shape == H_b.shape for T_b != T_d."""
    pytest.skip("not implemented yet")


def test_zero_gate_is_identity():
    """Force gate to 0 -> H == H_b."""
    pytest.skip("not implemented yet")


def test_gate_in_unit_interval():
    """sigmoid gate in (0,1)."""
    pytest.skip("not implemented yet")


def test_concat_fusion_shapes():
    """ConcatFusion output matches H_b shape."""
    pytest.skip("not implemented yet")


def test_gradients_reach_both_streams():
    """Gradients reach H_b and H_d."""
    pytest.skip("not implemented yet")

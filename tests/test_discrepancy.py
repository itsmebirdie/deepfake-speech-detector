"""Owner: Member B. Tests are specified but skipped until the module is implemented."""
import pytest


def test_identical_phase_gives_zero_residual():
    """X == Y_K -> cos=1*w, sin=0, d_psi=0."""
    pytest.skip("not implemented yet")


def test_wrap_invariance():
    """Adding 2*pi to the phase of Y_K leaves D unchanged."""
    pytest.skip("not implemented yet")


def test_confidence_weight_range():
    """0 <= w < 1 and monotone in A."""
    pytest.skip("not implemented yet")


def test_low_energy_bins_suppressed():
    """Near-silent bins contribute ~0 to D and to cons_per_sample."""
    pytest.skip("not implemented yet")


def test_raw_phase_mode():
    """mode='raw_phase' bypasses ULCP and returns observed-phase channels."""
    pytest.skip("not implemented yet")


def test_output_shape():
    """[B,3,F,N] (or [B,2,F,N] without IF channel)."""
    pytest.skip("not implemented yet")

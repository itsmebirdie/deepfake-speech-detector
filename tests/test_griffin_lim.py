"""Owner: Member B. Tests are specified but skipped until the module is implemented."""
import pytest


def test_gl_preserves_magnitude():
    """Returned STFT has |Y| == input magnitude."""
    pytest.skip("not implemented yet")


def test_gl_spectral_convergence_improves():
    """Spectral convergence decreases with more iterations on a real clip."""
    pytest.skip("not implemented yet")


def test_instantaneous_frequency_of_sinusoid():
    """IF of a pure tone is constant and equals the tone frequency."""
    pytest.skip("not implemented yet")


def test_wrap_to_pi_range():
    """Output always in (-pi, pi]."""
    pytest.skip("not implemented yet")

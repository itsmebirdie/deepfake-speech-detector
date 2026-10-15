"""Owner: Member C. Tests are specified but skipped until the module is implemented."""
import pytest


def test_total_loss_composition():
    """total == bin + beta*voc + mu*cons on synthetic inputs."""
    pytest.skip("not implemented yet")


def test_missing_terms_are_skipped():
    """Baseline-only outputs (no cons key) do not crash."""
    pytest.skip("not implemented yet")


def test_mu_zero_removes_cons_gradient():
    """mu=0 -> no gradient through cons term."""
    pytest.skip("not implemented yet")

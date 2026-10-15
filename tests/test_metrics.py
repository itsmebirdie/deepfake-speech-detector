"""Owner: Member A. Tests are specified but skipped until the module is implemented."""
import pytest


def test_eer_perfect_separation():
    """EER == 0 for perfectly separable scores."""
    pytest.skip("not implemented yet")


def test_eer_random_scores():
    """EER ~ 0.5 for random scores (tolerance)."""
    pytest.skip("not implemented yet")


def test_eer_matches_baseline_implementation():
    """Same scores -> same EER as the baseline repo's function."""
    pytest.skip("not implemented yet")


def test_auc_known_value():
    """AUC on a hand-built example equals the analytic value."""
    pytest.skip("not implemented yet")


def test_accuracy_threshold():
    """Accuracy at a given threshold on a toy set."""
    pytest.skip("not implemented yet")

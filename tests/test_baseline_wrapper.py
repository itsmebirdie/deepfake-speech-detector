"""Owner: Member C. Tests are specified but skipped until the module is implemented."""
import pytest


def test_logit_parity_with_baseline_repo():
    """Wrapper logits == baseline repo's own forward on the same clips."""
    pytest.skip("not implemented yet")


def test_forward_features_shape():
    """H_b is [B,T_b,C_b], documented in INTERFACES.md."""
    pytest.skip("not implemented yet")


def test_freeze_encoder_flag():
    """freeze_encoder=True -> no encoder param requires grad."""
    pytest.skip("not implemented yet")


def test_label_convention_mapping():
    """Repo 0/1 labels map correctly to baseline convention and back."""
    pytest.skip("not implemented yet")

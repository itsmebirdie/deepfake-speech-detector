"""Owner: Member C. Tests are specified but skipped until the module is implemented."""
import pytest


def test_forward_keys_and_shapes():
    """logits_bin [B,2], logits_voc [B,n_voc], cons_per_sample [B]."""
    pytest.skip("not implemented yet")


def test_phase_branch_disabled_equals_baseline():
    """enabled=false reproduces baseline logits."""
    pytest.skip("not implemented yet")


def test_backward_updates_new_modules():
    """After one step, ULCP/fusion/encoder params change; frozen ones do not."""
    pytest.skip("not implemented yet")


def test_overfit_single_batch():
    """Loss on one batch falls near zero (slow; marker)."""
    pytest.skip("not implemented yet")


def test_each_ablation_config_builds():
    """Every configs/ablation/*.yaml instantiates a model and runs a forward pass."""
    pytest.skip("not implemented yet")

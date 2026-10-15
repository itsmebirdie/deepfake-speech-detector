"""Owner: Member B. Tests are specified but skipped until the module is implemented."""
import pytest


def test_interface_keys():
    """Output has H_d and cons_per_sample with documented shapes."""
    pytest.skip("not implemented yet")


def test_batch_independence():
    """Result for sample i does not depend on other samples in the batch."""
    pytest.skip("not implemented yet")


def test_consistency_loss_masks_bona_fide():
    """consistency_loss ignores label==1 samples and handles all-fake batches."""
    pytest.skip("not implemented yet")

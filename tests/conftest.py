"""Shared fixtures (dummy waveforms, tiny configs). Add as modules come online."""
import pytest


@pytest.fixture
def dummy_wav():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    return torch.randn(2, 64600)

"""Shared fixtures (dummy waveforms, tiny configs). Add as modules come online."""
import pytest


@pytest.fixture
def dummy_wav():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    return torch.randn(2, 64600)


# ---------------------------------------------------------------------------------------------
# Fixtures for the model tests (Member C). They need the authors' baseline repo; tests that ask
# for `baseline_dir` are skipped (with a reason) when it cannot be found.
# ---------------------------------------------------------------------------------------------
import os  # noqa: E402
from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASED_WEIGHTS = REPO_ROOT / "weights" / "baseline_pretrained.pth"


@pytest.fixture(scope="session")
def baseline_dir():
    """Folder of the authors' repo: $PHASECON_BASELINE_DIR, the submodule, or ../baseline-repo."""
    candidates = [
        os.environ.get("PHASECON_BASELINE_DIR"),
        REPO_ROOT / "third_party" / "Synthetic-Voice-Detection-Vocoder-Artifacts",
        REPO_ROOT.parent / "baseline-repo",
    ]
    for c in candidates:
        if c and (Path(c) / "model.py").is_file():
            return str(c)
    pytest.skip("authors' baseline repo not found: run `make baseline` or set PHASECON_BASELINE_DIR")


@pytest.fixture(scope="session")
def random_baseline_weights(baseline_dir, tmp_path_factory):
    """A state dict for the baseline RawNet with random (but fixed) weights and non-trivial
    BatchNorm statistics. Stands in for the released weights when those are not available."""
    import copy
    import importlib.util

    import torch
    import yaml

    spec = importlib.util.spec_from_file_location("ref_baseline_model", os.path.join(baseline_dir, "model.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with open(os.path.join(baseline_dir, "model_config_RawNet.yaml")) as f:
        cfg = copy.deepcopy(yaml.safe_load(f)["model"])
    torch.manual_seed(1234)
    net = module.RawNet(cfg, "cpu")
    net.train()
    with torch.no_grad():  # a few train-mode passes give the BatchNorm layers real running statistics
        for _ in range(2):
            net(torch.randn(4, 9600) * 0.1)
    path = tmp_path_factory.mktemp("weights") / "random_baseline.pth"
    torch.save(net.state_dict(), path)
    return str(path)

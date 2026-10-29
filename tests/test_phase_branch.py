"""Owner: Member B. Tests for models/phase_branch.py (PR-21); the consistency-loss test is PR-22."""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

from phasecon.models.phase_branch import PhaseBranch  # noqa: E402

T = 12_000  # 0.5 s at 24 kHz keeps the tests fast; H_d is [B, T_d, C_d] for any length


def phase_cfg():
    """The real `model.phase_branch` block from configs/model/phasecon.yaml."""
    path = Path(__file__).parents[1] / "configs/model/phasecon.yaml"
    return yaml.safe_load(path.read_text())["model"]["phase_branch"]


def test_interface_keys():
    """Output has H_d and cons_per_sample with documented shapes."""
    torch.manual_seed(0)
    cfg = phase_cfg()
    branch = PhaseBranch(cfg)
    out = branch(torch.randn(3, T) * 0.1)
    t_d, c_d = cfg["encoder"]["t_out"], cfg["encoder"]["out_dim"]
    assert set(out) == {"H_d", "cons_per_sample"}
    assert out["H_d"].shape == (3, t_d, c_d) and out["H_d"].dtype == torch.float32
    assert out["cons_per_sample"].shape == (3,)
    assert torch.isfinite(out["H_d"]).all() and (out["cons_per_sample"] >= 0).all()

    out["H_d"].sum().backward()  # gradients reach ULCP through the discrepancy tensor
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.ulcp.parameters())

    cfg["return_delta_phi"] = True
    out = PhaseBranch(cfg)(torch.randn(2, T) * 0.1)
    assert [d.shape[1] for d in out["delta_phi"]] == [n // 2 + 1 for n, _, _ in cfg["stft_resolutions"]]

    cfg["discrepancy"]["mode"] = "raw_phase"  # ablation (i): no ULCP, no consistency term
    out = PhaseBranch(cfg)(torch.randn(2, T) * 0.1)
    assert out["H_d"].shape == (2, t_d, c_d) and torch.equal(out["cons_per_sample"], torch.zeros(2))


def test_batch_independence():
    """Result for sample i does not depend on other samples in the batch."""
    torch.manual_seed(1)
    branch = PhaseBranch(phase_cfg()).train()  # train mode: a BatchNorm would mix clips here
    wav = torch.randn(4, T) * 0.1
    full = branch(wav)
    other = wav.clone()
    other[1:] = torch.randn(3, T)  # change every other clip, louder
    for out in (branch(wav[:1]), branch(other)):
        assert torch.allclose(out["H_d"][0], full["H_d"][0], atol=1e-5)
        assert torch.allclose(out["cons_per_sample"][0], full["cons_per_sample"][0], atol=1e-6)


def test_consistency_loss_masks_bona_fide():
    """consistency_loss ignores label==1 samples and handles all-fake batches."""
    from phasecon.losses.consistency import consistency_loss

    cons = torch.tensor([0.2, 5.0, 0.4, 9.0], requires_grad=True)
    label = torch.tensor([0, 1, 0, 1])
    loss = consistency_loss(cons, label)
    assert loss.item() == pytest.approx(0.3)  # mean of the two real clips; fakes (5.0, 9.0) ignored
    loss.backward()
    assert cons.grad.tolist() == pytest.approx([0.5, 0.0, 0.5, 0.0])  # gradient only on real clips

    cons = torch.tensor([3.0, 7.0], requires_grad=True)
    loss = consistency_loss(cons, torch.tensor([1, 1]))  # all fake
    assert loss.item() == 0.0 and torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    assert cons.grad.tolist() == [0.0, 0.0]  # graph intact, no NaN

    with pytest.raises(ValueError):
        consistency_loss(torch.zeros(3), torch.zeros(2))

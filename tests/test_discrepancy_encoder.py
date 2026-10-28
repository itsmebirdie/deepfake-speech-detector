"""Owner: Member B. Tests for models/discrepancy_encoder.py (PR-19)."""
import pytest

torch = pytest.importorskip("torch")

from phasecon.models.discrepancy_encoder import DEFAULT_T_OUT, DiscrepancyEncoder  # noqa: E402

RES = [(512, 128), (1024, 256), (2048, 512)]  # (n_fft, hop)


def fake_D(batch, num_samples, num_res, c_in=3):
    """Random tensors with the real per-resolution shapes [B, C_in, n_fft//2+1, T//hop+1]."""
    return [torch.randn(batch, c_in, n // 2 + 1, num_samples // h + 1) for n, h in RES[:num_res]]


def test_output_shape_fixed_td():
    """H_d is [B,T_d,out_dim] for any input length / resolution count."""
    torch.manual_seed(0)
    for num_res in (1, 2, 3):
        for c_in in (2, 3):
            enc = DiscrepancyEncoder(c_in, [8, 16], out_dim=32, num_resolutions=num_res, t_out=43).eval()
            for num_samples in (12_000, 96_000):
                H = enc(fake_D(2, num_samples, num_res, c_in))
                assert H.shape == (2, 43, 32)
                assert torch.isfinite(H).all()
    with pytest.raises(ValueError):
        DiscrepancyEncoder(3, [8], out_dim=32, num_resolutions=3)(fake_D(1, 12_000, 2))


def test_gradients_flow():
    """Gradient reaches all resolutions' branches."""
    torch.manual_seed(1)
    enc = DiscrepancyEncoder(3, [8, 16], out_dim=32, num_resolutions=3, t_out=43)
    enc(fake_D(2, 12_000, 3)).pow(2).mean().backward()
    for branch in enc.branches:
        first_conv = branch[0]
        assert first_conv.weight.grad is not None and first_conv.weight.grad.abs().sum() > 0
    assert enc.proj.weight.grad.abs().sum() > 0


def test_t_out_matches_rawnet2_sequence_length():
    """Config t_out equals T_b = 43 for 96,000 samples (PR-08), so fusion sees equal-length sequences."""
    import yaml
    from pathlib import Path

    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs/model/phasecon.yaml").read_text())
    assert cfg["model"]["phase_branch"]["encoder"]["t_out"] == DEFAULT_T_OUT == 43
    assert DiscrepancyEncoder(3, [8], 16, 1).t_out == 43

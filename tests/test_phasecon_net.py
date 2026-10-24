"""Owner: Member C (PR-18; ablation configs of PR-28). Tests for models/phasecon_net.py and build_model.

All tests here use the DummyPhaseBranch (`impl: dummy`) because Member B's PhaseBranch (PR-21) is not
merged. The tests that need the real branch (ULCP / discrepancy encoder updating, overfitting the
full model, B's ablation configs) stay skipped with the reason below.
"""
import warnings
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("yaml")

from phasecon.losses.combined import total_loss  # noqa: E402
from phasecon.models import build_model  # noqa: E402
from phasecon.models.baseline.rawnet2_encoder import NATIVE_NUM_SAMPLES  # noqa: E402
from phasecon.models.dummy_phase_branch import DummyPhaseBranch  # noqa: E402
from phasecon.models.fusion import ConcatFusion, GatedCrossAttentionFusion  # noqa: E402
from phasecon.utils.config import load_many  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "configs"
WAV_LEN = 12_000  # T_b = 5: short enough for fast CPU tests
NEEDS_REAL_BRANCH = "needs Member B's real PhaseBranch (PR-21/22), not merged yet"


def make_cfg(baseline_dir, weights, model_file="model/phasecon.yaml", extra_files=(), overrides=()):
    files = [CFG / "default.yaml", CFG / "data" / "librisevoc.yaml", CFG / model_file, *extra_files]
    ov = [
        f"model.baseline.repo_dir={baseline_dir}",
        f"model.baseline.weights={weights}" if weights else "model.baseline.weights=null",
        "model.phase_branch.impl=dummy",
        "model.phase_branch.encoder.t_out=8",
        "model.fusion.dim=16",
        "audio.sample_rate=24000",  # the baseline's native audio; the repo default (16 kHz) makes build_model warn
        "audio.num_samples=96000",
        *overrides,
    ]
    return load_many([str(f) for f in files], ov)


def batch_for(n=4):
    torch.manual_seed(0)
    vid = torch.tensor([0, 1, 4, 0, 6, 2])[:n]
    return torch.randn(n, WAV_LEN) * 0.1, {"label": (vid > 0).long(), "vocoder_id": vid}


@pytest.fixture(scope="module")
def phasecon_model(baseline_dir, random_baseline_weights):
    return build_model(make_cfg(baseline_dir, random_baseline_weights)).eval()


@pytest.fixture(scope="module")
def baseline_only_model(baseline_dir, random_baseline_weights):
    cfg = make_cfg(baseline_dir, random_baseline_weights, "model/baseline_only.yaml")
    return build_model(cfg).eval()


def test_forward_keys_and_shapes(phasecon_model):
    """logits_bin [B,2], logits_voc [B,n_voc], cons_per_sample [B]."""
    wav, _ = batch_for(4)
    with torch.no_grad():
        out = phasecon_model(wav)
    assert set(out) == {"logits_bin", "logits_voc", "cons_per_sample"}
    assert out["logits_bin"].shape == (4, 2)
    assert out["logits_voc"].shape == (4, 7)
    assert out["cons_per_sample"].shape == (4,)
    assert phasecon_model.phase_branch_is_dummy and isinstance(phasecon_model.fusion, GatedCrossAttentionFusion)
    assert isinstance(phasecon_model.phase_branch, DummyPhaseBranch)
    assert phasecon_model.phase_branch.t_d == 8  # model.phase_branch.encoder.t_out; T_b is 5, so T_d != T_b


def test_forward_works_at_native_input_length(phasecon_model):
    with torch.no_grad():
        out = phasecon_model(torch.randn(2, NATIVE_NUM_SAMPLES) * 0.1)
    assert out["logits_bin"].shape == (2, 2)


def test_phase_branch_disabled_equals_baseline(baseline_only_model):
    """enabled=false reproduces baseline logits."""
    model = baseline_only_model
    assert model.phase_branch is None and model.fusion is None
    wav, _ = batch_for(4)
    with torch.no_grad():
        out = model(wav)
        ref = model.baseline(wav)  # the wrapper, which test_baseline_wrapper checks against the baseline repo
    assert set(out) == {"logits_bin", "logits_voc"}  # no cons key without a phase branch
    # our heads return raw logits; the baseline's outputs are their log-softmax (same up to a constant)
    torch.testing.assert_close(torch.log_softmax(out["logits_bin"], 1), ref["logits_bin"], rtol=0, atol=1e-6)
    torch.testing.assert_close(torch.log_softmax(out["logits_voc"], 1), ref["logits_voc"], rtol=0, atol=1e-6)
    assert torch.equal(out["logits_bin"].argmax(1), ref["logits_bin"].argmax(1))
    s_new = out["logits_bin"][:, 1] - out["logits_bin"][:, 0]
    s_ref = ref["logits_bin"][:, 1] - ref["logits_bin"][:, 0]
    torch.testing.assert_close(s_new, s_ref, rtol=0, atol=1e-5)  # the evaluator's score


def test_baseline_only_name_rejects_enabled_phase_branch(baseline_dir, random_baseline_weights):
    cfg = make_cfg(baseline_dir, random_baseline_weights, "model/baseline_only.yaml",
                   overrides=["model.phase_branch.enabled=true"])
    with pytest.raises(ValueError):
        build_model(cfg)


def test_build_model_validation(baseline_dir, random_baseline_weights):
    with pytest.raises(KeyError):
        build_model({})
    cfg = make_cfg(baseline_dir, random_baseline_weights)
    cfg["model"]["name"] = "nope"
    with pytest.raises(ValueError):
        build_model(cfg)
    for bad in ("model.heads.pooling=max", "model.phase_branch.impl=fake", "model.fusion.type=sum"):
        with pytest.raises(ValueError):
            build_model(make_cfg(baseline_dir, random_baseline_weights, overrides=[bad]))
    with pytest.raises(ValueError):  # gradient reversal needs the vocoder head
        build_model(make_cfg(baseline_dir, random_baseline_weights,
                             overrides=["model.heads.vocoder_head=false", "model.heads.gradient_reversal=true"]))


def test_audio_config_mismatch_warns(baseline_dir, random_baseline_weights):
    cfg = make_cfg(baseline_dir, random_baseline_weights, "model/baseline_only.yaml")
    cfg["audio"] = {"sample_rate": 16000, "num_samples": 64000}  # the repo default, pending the INTERFACES.md decision
    with pytest.warns(RuntimeWarning, match="24000"):
        build_model(cfg)
    cfg["audio"] = {"sample_rate": 24000, "num_samples": 96000}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_model(cfg)


def test_auto_impl_falls_back_loudly_while_phase_branch_is_a_stub(baseline_dir, random_baseline_weights):
    cfg = make_cfg(baseline_dir, random_baseline_weights, overrides=["model.phase_branch.impl=auto"])
    try:
        from phasecon.models.phase_branch import PhaseBranch

        PhaseBranch({})
        pytest.skip("PhaseBranch is implemented now; the fallback is no longer reachable")
    except NotImplementedError:
        pass
    with pytest.warns(RuntimeWarning, match="RANDOM"):
        model = build_model(cfg)
    assert model.phase_branch_is_dummy
    cfg["model"]["phase_branch"]["impl"] = "real"  # explicit 'real' must not fall back
    with pytest.raises(NotImplementedError):
        build_model(cfg)


def test_pooling_options(baseline_dir, random_baseline_weights):
    wav, _ = batch_for(2)
    last = build_model(make_cfg(baseline_dir, random_baseline_weights, "model/baseline_only.yaml")).eval()
    mean = build_model(make_cfg(baseline_dir, random_baseline_weights, "model/baseline_only.yaml",
                                overrides=["model.heads.pooling=mean"])).eval()
    with torch.no_grad():
        assert not torch.allclose(last(wav)["logits_bin"], mean(wav)["logits_bin"])


def train_step(model, lr_encoder=1e-3, lr_new=1e-3, mu=0.0):
    """One optimizer step on a fixed batch; returns the loss."""
    wav, batch = batch_for(4)
    opt = torch.optim.Adam(model.param_groups(lr_encoder, lr_new, 0.0))
    model.train()
    loss, _ = total_loss(model(wav), batch, beta=0.5, mu=mu)
    loss.backward()
    opt.step()
    return loss


def snapshot(module):
    return {n: p.detach().clone() for n, p in module.named_parameters()}


def changed(before, module):
    return {n for n, p in module.named_parameters() if not torch.equal(before[n], p.detach())}


def test_param_groups_partition_the_trainable_parameters(baseline_dir, random_baseline_weights):
    model = build_model(make_cfg(baseline_dir, random_baseline_weights))
    groups = model.param_groups(1e-5, 1e-4, 1e-4)
    assert [g["name"] for g in groups] == ["encoder", "new"]
    assert [g["lr"] for g in groups] == [1e-5, 1e-4] and all(g["weight_decay"] == 1e-4 for g in groups)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids))  # no parameter twice
    unused = {id(p) for p in model.baseline.head_parameters()}  # the baseline's own heads are not used
    assert not unused & set(ids)
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert set(ids) == trainable - unused
    frozen = build_model(make_cfg(baseline_dir, random_baseline_weights, overrides=["model.baseline.freeze_encoder=true"]))
    assert [g["name"] for g in frozen.param_groups(1e-5, 1e-4, 0.0)] == ["new"]


def test_backward_updates_new_modules_with_dummy_branch(baseline_dir, random_baseline_weights):
    """After one step the fusion, the heads and the encoder change; a frozen encoder does not.
    (The ULCP / discrepancy-encoder part of the original test needs the real branch.)"""
    model = build_model(make_cfg(baseline_dir, random_baseline_weights))
    before = snapshot(model)
    loss = train_step(model)
    assert torch.isfinite(loss)
    for part in (model.fusion, model.binary_head, model.vocoder_head, model.baseline):
        sub = {n for n, p in model.named_parameters() if any(p is q for q in part.parameters())}
        assert changed(before, model) & sub, f"{type(part).__name__} did not change"
    # the baseline's own heads are unused: untouched
    own_heads = {n for n, p in model.named_parameters() if any(p is q for q in model.baseline.head_parameters())}
    assert not changed(before, model) & own_heads

    frozen = build_model(make_cfg(baseline_dir, random_baseline_weights, overrides=["model.baseline.freeze_encoder=true"]))
    before = snapshot(frozen)
    train_step(frozen)
    moved = changed(before, frozen)
    assert moved and not any(n.startswith("baseline.") for n in moved)
    assert any(n.startswith("fusion.") for n in moved) and any(n.startswith("binary_head.") for n in moved)


def test_phase_stream_runs_in_float32_under_autocast(baseline_dir, random_baseline_weights):
    seen = {}

    class Probe(torch.nn.Module):
        def forward(self, wav):
            seen["dtype"] = wav.dtype
            seen["autocast"] = torch.is_autocast_enabled("cpu")
            return {"H_d": torch.zeros(wav.shape[0], 8, 128), "cons_per_sample": torch.zeros(wav.shape[0])}

    model = build_model(make_cfg(baseline_dir, random_baseline_weights)).eval()
    model.phase_branch = Probe()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(torch.randn(2, WAV_LEN) * 0.1)
    assert seen == {"dtype": torch.float32, "autocast": False}
    assert out["logits_bin"].shape == (2, 2)


def test_backward_updates_new_modules():
    """After one step, ULCP/fusion/encoder params change; frozen ones do not."""
    pytest.skip(NEEDS_REAL_BRANCH + " (covered for the dummy branch above; PR-23 adds the real one)")


def test_overfit_single_batch():
    """Loss on one batch falls near zero (slow; marker)."""
    pytest.skip(NEEDS_REAL_BRANCH + " (the trainer's overfit_batch is tested on the dummy branch in test_trainer.py)")


ABLATIONS = sorted((CFG / "ablation").glob("*.yaml"))
C_ABLATIONS = {"a3_concat_fusion", "a4_no_cons_loss", "a5_gradient_reversal", "a7_frozen_encoder"}


def _ablation_params():
    for path in ABLATIONS:
        marks = [] if path.stem in C_ABLATIONS else [pytest.mark.skip(reason=NEEDS_REAL_BRANCH + " (PR-27)")]
        yield pytest.param(path, id=path.stem, marks=marks)


@pytest.mark.parametrize("path", list(_ablation_params()))
def test_each_ablation_config_builds(path, baseline_dir, random_baseline_weights):
    """Every configs/ablation/*.yaml instantiates a model and runs a forward + backward pass.
    Member C's four (a3, a4, a5, a7) run here on the dummy branch; Member B's need the real one."""
    cfg = make_cfg(baseline_dir, random_baseline_weights, "model/baseline_only.yaml", extra_files=[path])
    cfg["model"]["phase_branch"]["impl"] = "dummy"
    model = build_model(cfg)
    assert model.phase_enabled  # ablation files inherit phasecon.yaml
    wav, batch = batch_for(4)
    model.train()
    out = model(wav)
    mu = float(cfg.get("train", {}).get("loss", {}).get("mu", 0.0))
    loss, parts = total_loss(out, batch, beta=0.5, mu=0.0)  # mu forced to 0: the consistency loss is a stub
    loss.backward()
    assert torch.isfinite(loss)
    name = path.stem
    if name == "a3_concat_fusion":
        assert isinstance(model.fusion, ConcatFusion)
        assert model.fusion.proj.weight.grad is not None
    else:
        assert isinstance(model.fusion, GatedCrossAttentionFusion)
    if name == "a4_no_cons_loss":
        assert mu == 0.0 and cfg["train"]["loss"]["mu"] == 0.0  # the config really sets mu = 0
        assert "cons" not in total_loss(out, batch, beta=0.5, mu=mu)[1]
    if name == "a5_gradient_reversal":
        assert model.vocoder_head.gradient_reversal
        assert out["logits_voc"].shape == (4, 7)
    else:
        assert not model.vocoder_head.gradient_reversal
    if name == "a7_frozen_encoder":
        assert model.baseline.freeze_encoder
        assert all(p.grad is None for p in model.baseline.encoder_parameters())
        assert [g["name"] for g in model.param_groups(1e-5, 1e-4, 0.0)] == ["new"]
    else:
        assert any(p.grad is not None for p in model.baseline.encoder_parameters())
    assert any(p.grad is not None for p in model.fusion.parameters())

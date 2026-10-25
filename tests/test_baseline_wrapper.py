"""Owner: Member C (PR-08). Tests for models/baseline/rawnet2_encoder.py.

They need the authors' baseline repo (see the `baseline_dir` fixture in conftest.py). The parity
tests run twice when the released weights exist at weights/baseline_pretrained.pth; otherwise they
use a random-weight stand-in, which checks the wrapper but not the released checkpoint itself.
"""
import copy
import importlib.util
import os

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

from phasecon.models.baseline.rawnet2_encoder import (  # noqa: E402
    NATIVE_NUM_SAMPLES,
    BaselineRawNet2,
)

from .conftest import RELEASED_WEIGHTS  # noqa: E402


@pytest.fixture(params=["random", "released"])
def weights(request, random_baseline_weights):
    if request.param == "random":
        return random_baseline_weights
    if not RELEASED_WEIGHTS.is_file():
        pytest.skip("released weights not found at weights/baseline_pretrained.pth (run `make weights`)")
    return str(RELEASED_WEIGHTS)


def reference_model(baseline_dir, weights_path):
    """The authors' RawNet built straight from their code, independently of the wrapper."""
    spec = importlib.util.spec_from_file_location("ref_baseline_model_2", os.path.join(baseline_dir, "model.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with open(os.path.join(baseline_dir, "model_config_RawNet.yaml")) as f:
        cfg = copy.deepcopy(yaml.safe_load(f)["model"])
    net = module.RawNet(cfg, "cpu")
    net.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return net.eval()


def make_wrapper(baseline_dir, weights_path, freeze=False):
    return BaselineRawNet2(baseline_dir, weights_path, freeze_encoder=freeze, num_vocoder_classes=7)


def test_logit_parity_with_baseline_repo(baseline_dir, weights):
    """Wrapper logits == baseline repo's own forward on the same clips."""
    ref = reference_model(baseline_dir, weights)
    wrapper = make_wrapper(baseline_dir, weights).eval()
    torch.manual_seed(0)
    for batch, length in [(3, 12_000), (2, NATIVE_NUM_SAMPLES)]:
        wav = torch.randn(batch, length) * 0.1
        with torch.no_grad():
            ref_bin, ref_voc = ref(wav)
            out = wrapper(wav)
        # binary: the baseline's [synthetic, bona fide] columns come back as [bona fide, synthetic]
        assert torch.equal(out["logits_bin"][:, 0], ref_bin[:, 1])
        assert torch.equal(out["logits_bin"][:, 1], ref_bin[:, 0])
        assert torch.equal(out["logits_voc"], ref_voc)
        # the evaluator's score equals the one scripts/eval_baseline.py computes
        score = out["logits_bin"][:, 1] - out["logits_bin"][:, 0]
        assert torch.equal(score, ref_bin[:, 0] - ref_bin[:, 1])


def test_forward_features_shape(baseline_dir, random_baseline_weights):
    """H_b is [B,T_b,C_b], documented in INTERFACES.md (T_b = 43 and C_b = 1024 for 96,000 samples)."""
    wrapper = make_wrapper(baseline_dir, random_baseline_weights).eval()
    assert wrapper.feature_dim == 1024
    with torch.no_grad():
        h = wrapper.forward_features(torch.randn(2, NATIVE_NUM_SAMPLES) * 0.1)
        assert h.shape == (2, 43, 1024)
        assert wrapper.feature_length(NATIVE_NUM_SAMPLES) == 43
        # the formula matches the real network for every length we use or document
        for n in (wrapper.min_num_samples, 4_000, 9_600, 12_000, 64_000, 96_000):
            assert wrapper.forward_features(torch.randn(2, n) * 0.1).shape[1:] == (
                wrapper.feature_length(n),
                1024,
            )
        assert wrapper.feature_length(64_000) == 28
        # the features come from the same pass as the logits, and the heads read the last step
        wav = torch.randn(2, 12_000) * 0.1
        out = wrapper(wav)
        assert torch.equal(out["H_b"], wrapper.forward_features(wav))
        heads = wrapper.pretrained_heads()["binary"]
        raw = (out["H_b"][:, -1, :] @ heads["fc1.weight"].T + heads["fc1.bias"]) @ heads["fc2.weight"].T + heads["fc2.bias"]
        torch.testing.assert_close(torch.log_softmax(raw, dim=1), out["logits_bin"], rtol=0, atol=1e-5)


def test_short_or_malformed_input_is_rejected(baseline_dir, random_baseline_weights):
    wrapper = make_wrapper(baseline_dir, random_baseline_weights).eval()
    with pytest.raises(ValueError):
        wrapper(torch.randn(2, wrapper.min_num_samples - 1))
    with pytest.raises(ValueError):
        wrapper(torch.randn(2, 1, 12_000))
    with pytest.raises(TypeError):
        wrapper(torch.zeros(2, 12_000, dtype=torch.long))
    # exactly the minimum works and gives a single time step
    with torch.no_grad():
        assert wrapper.forward_features(torch.randn(1, wrapper.min_num_samples)).shape == (1, 1, 1024)


def test_freeze_encoder_flag(baseline_dir, random_baseline_weights):
    """freeze_encoder=True -> no encoder param requires grad (the baseline's own heads stay trainable)."""
    frozen = make_wrapper(baseline_dir, random_baseline_weights, freeze=True)
    assert frozen.freeze_encoder
    enc = list(frozen.encoder_parameters())
    assert len(enc) > 50 and not any(p.requires_grad for p in enc)
    heads = list(frozen.head_parameters())
    assert len(heads) == 8 and all(p.requires_grad for p in heads)

    open_ = make_wrapper(baseline_dir, random_baseline_weights, freeze=False)
    assert all(p.requires_grad for p in open_.parameters())

    # the frozen part stays in eval mode, so BatchNorm statistics do not move in a training step
    frozen.train()
    assert not frozen.net.first_bn.training and not frozen.net.gru.training
    before = frozen.net.first_bn.running_mean.clone()
    frozen(torch.randn(4, 12_000))
    assert torch.equal(before, frozen.net.first_bn.running_mean)
    # ... whereas an unfrozen model in train mode does update them
    open_.train()
    before = open_.net.first_bn.running_mean.clone()
    open_(torch.randn(4, 12_000))
    assert not torch.equal(before, open_.net.first_bn.running_mean)

    # unfreezing again restores gradients, and only the heads receive gradient while frozen
    frozen.set_freeze_encoder(False)
    assert all(p.requires_grad for p in frozen.parameters())
    only_heads = make_wrapper(baseline_dir, random_baseline_weights, freeze=True)
    out = only_heads(torch.randn(2, 12_000))
    (out["logits_bin"].sum() + out["logits_voc"].sum()).backward()  # both heads in the loss
    assert all(p.grad is None for p in only_heads.encoder_parameters())
    assert all(p.grad is not None for p in only_heads.head_parameters())


def test_label_convention_mapping(baseline_dir, random_baseline_weights):
    """Repo 0/1 labels map correctly to baseline convention and back."""
    label = torch.tensor([0, 1, 1, 0])  # repo: 0 bona fide, 1 synthetic
    base = BaselineRawNet2.repo_label_to_baseline(label)
    assert base.tolist() == [1, 0, 0, 1]  # baseline target is (vocoder == 0): 1 = bona fide
    assert torch.equal(BaselineRawNet2.baseline_label_to_repo(base), label)

    # end to end: make the baseline's binary head say "bona fide" (its index 1), and check that
    # the wrapper's argmax is the repo's label 0
    wrapper = make_wrapper(baseline_dir, random_baseline_weights).eval()
    with torch.no_grad():
        wrapper.net.fc2_binary_gru.weight.zero_()
        wrapper.net.fc2_binary_gru.bias.copy_(torch.tensor([-5.0, 5.0]))  # [synthetic, bona fide]
        out = wrapper(torch.randn(3, 12_000))
    assert out["logits_bin"].argmax(1).tolist() == [0, 0, 0]
    score = out["logits_bin"][:, 1] - out["logits_bin"][:, 0]
    assert (score < 0).all()  # negative score = bona fide, the INTERFACES.md convention

    # the exported head weights follow the same swap
    head = wrapper.pretrained_heads()["binary"]
    assert torch.equal(head["fc2.bias"], torch.tensor([5.0, -5.0]))


def test_building_twice_does_not_corrupt_the_config(baseline_dir, random_baseline_weights):
    """RawNet.__init__ edits its config dict in place; two wrappers must still be identical."""
    a = make_wrapper(baseline_dir, random_baseline_weights).eval()
    b = make_wrapper(baseline_dir, random_baseline_weights).eval()
    assert a.state_dict().keys() == b.state_dict().keys()
    wav = torch.randn(2, 12_000) * 0.1
    with torch.no_grad():
        assert torch.equal(a(wav)["logits_bin"], b(wav)["logits_bin"])


def test_bad_arguments(baseline_dir, tmp_path):
    with pytest.raises(ValueError):
        BaselineRawNet2(baseline_dir, None, False, num_vocoder_classes=5)
    with pytest.raises(FileNotFoundError):
        BaselineRawNet2(str(tmp_path), None, False, 7)  # no model.py there
    with pytest.raises(FileNotFoundError):
        BaselineRawNet2(baseline_dir, str(tmp_path / "missing.pth"), False, 7)
    bad = tmp_path / "bad.pth"
    torch.save({"not": torch.zeros(1)}, bad)
    with pytest.raises(RuntimeError):
        BaselineRawNet2(baseline_dir, str(bad), False, 7)


def test_works_under_inference_mode_then_training(baseline_dir, random_baseline_weights):
    """The evaluator runs under inference_mode; a later training step must still work."""
    wrapper = make_wrapper(baseline_dir, random_baseline_weights)
    wav = torch.randn(2, 12_000) * 0.1
    with torch.inference_mode():
        wrapper.eval()
        wrapper(wav)
    wrapper.train()
    out = wrapper(wav)
    (out["logits_bin"].sum() + out["logits_voc"].sum()).backward()
    # Quirk of the authors' code (left untouched): Residual_block.forward overwrites the result of
    # bn1 with conv1(x), so the 10 `.bn1.` parameters in blocks 1-5 never get a gradient.
    no_grad = [n for n, p in wrapper.named_parameters() if p.grad is None]
    assert no_grad and all(".bn1." in n for n in no_grad)
    assert len(no_grad) == 10

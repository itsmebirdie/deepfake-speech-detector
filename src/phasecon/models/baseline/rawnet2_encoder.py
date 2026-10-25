"""Wrapper around the authors' released RawNet2 (+ vocoder-ID) model. Owner: Member C (PR-08).

The baseline repo lives in third_party/ (git submodule). Its code is NOT copied into src/: the
authors' `model.py` is loaded from the repo folder by file path, under a private module name, so
nothing is added to the global `sys.path` or registered as the top-level module `model`.

Facts about the baseline that this wrapper relies on (docs/DATASET_NOTES.md section 6, re-checked
against the code at commit 74b8527):

* Input: 24 kHz mono, 4 s = 96,000 samples. The Sinc front-end is built for 24 kHz.
* After the residual blocks the features are [B, C, T]; a 3-layer GRU turns them into [B, T, 1024]
  and the heads read ONLY the last time step (`x[:, -1, :]`).
* Both heads are `Linear -> Linear` (no activation between) followed by LogSoftmax.
* Binary head: index 0 = synthetic, index 1 = bona fide (training target is `y == 0`).
  Vocoder head: 7 classes, 0 = bona fide, 1..6 = vocoders (same order as `data.vocoders`).
* `RawNet.__init__` rewrites `d_args['filts'][2][0]` IN PLACE, so every build must get its own copy
  of the config dict (done below).
* `SincConv` has no parameters; it rebuilds its filters in every forward on the device given at
  construction. The wrapper keeps that device in step with the input.
* Quirk (left as is, the baseline repo is not modified): `Residual_block.forward` overwrites the
  output of its `bn1` layer with `conv1(x)`, so the `bn1` weight/bias of blocks 1-5 (10 tensors)
  never receive a gradient. They do not affect the output.
"""
import copy
import importlib.util
import os
import sys
from typing import Dict, Iterator, Optional

import torch
import torch.nn as nn
import yaml

NATIVE_SAMPLE_RATE = 24_000
NATIVE_NUM_SAMPLES = 96_000  # 4 s at 24 kHz: the length the released weights were trained on
BASELINE_VOCODER_CLASSES = 7  # `fc2_multi_gru` is hard-coded to 7 outputs in the baseline's model.py
_NUM_POOL_STAGES = 7  # one max-pool after the Sinc front-end, one in each of the 6 residual blocks
_POOL_FACTOR = 3
_HEAD_MODULES = ("fc1_binary_gru", "fc2_binary_gru", "fc1_multi_gru", "fc2_multi_gru")

_MODULE_CACHE: Dict[str, object] = {}


def _load_baseline_module(repo_dir: str):
    """Import the authors' `model.py` by path (cached). Does not leave anything on `sys.path`."""
    repo_dir = os.path.abspath(repo_dir)
    path = os.path.join(repo_dir, "model.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"baseline model.py not found at {path}. Run `make baseline` (adds the submodule under "
            "third_party/) or point model.baseline.repo_dir at a clone of the authors' repo."
        )
    if path in _MODULE_CACHE:
        return _MODULE_CACHE[path]
    spec = importlib.util.spec_from_file_location("phasecon_baseline_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, repo_dir)  # only while the file executes, in case it imports a sibling file
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(repo_dir)
        except ValueError:
            pass
    _MODULE_CACHE[path] = module
    return module


def _is_head_param(name: str) -> bool:
    """`name` is a parameter name inside the baseline RawNet (e.g. 'fc2_binary_gru.weight')."""
    return name.split(".")[0] in _HEAD_MODULES


class BaselineRawNet2(nn.Module):
    """Loads released weights and exposes the intermediate feature sequence.

    forward_features(wav [B,T]) -> H_b [B, T_b, C_b]   (GRU output at every time step, BEFORE pooling)
    forward(wav) -> {'logits_bin': [B,2], 'logits_voc': [B,7], 'H_b': H_b}

    Conventions (all outputs follow this repo, not the baseline):
    * `logits_bin` is [bona fide, synthetic] (label 0 / 1): the baseline's two columns, swapped.
      The values are the baseline's own log-probabilities, so they are bit-identical to its output.
      Log-probabilities are logits up to a per-row constant: cross-entropy and the evaluator's
      score (`logits_bin[:,1] - logits_bin[:,0]`) are unaffected by that constant.
    * `logits_voc` is the baseline's vocoder output unchanged (class 0 bona fide, 1..6 vocoders).
    * C_b = 1024 (GRU hidden size); T_b depends on the input length, see `feature_length`.
      T_b = 43 for the native 96,000 samples.
    * The baseline pools by taking the LAST time step: `H_b[:, -1, :]`.

    The heads and everything else in the baseline's forward run as the authors wrote them, so with
    identical weights the outputs equal the baseline repo's on identical clips. `forward_features`
    runs the same forward pass and returns the GRU sequence captured with a forward hook.

    freeze_encoder: every parameter except the four head linears gets requires_grad=False, and the
    frozen part stays in eval mode (BatchNorm statistics do not move) even after `.train()`.
    """

    def __init__(
        self,
        repo_dir: str,
        weights: Optional[str],
        freeze_encoder: bool,
        num_vocoder_classes: int = BASELINE_VOCODER_CLASSES,
    ):
        super().__init__()
        if num_vocoder_classes != BASELINE_VOCODER_CLASSES:
            raise ValueError(
                f"the baseline's vocoder head has {BASELINE_VOCODER_CLASSES} outputs "
                f"(bona fide + 6 vocoders); got num_vocoder_classes={num_vocoder_classes}"
            )
        module = _load_baseline_module(repo_dir)
        cfg_path = os.path.join(os.path.abspath(repo_dir), "model_config_RawNet.yaml")
        with open(cfg_path) as f:
            net_cfg = copy.deepcopy(yaml.safe_load(f)["model"])  # RawNet mutates it; never share
        self.net = module.RawNet(net_cfg, "cpu")
        self.num_vocoder_classes = num_vocoder_classes
        if weights:
            self._load_weights(weights)
        self.freeze_encoder = False
        self.set_freeze_encoder(bool(freeze_encoder))

    # ---- construction helpers -------------------------------------------------------------
    def _load_weights(self, weights: str) -> None:
        if not os.path.isfile(weights):
            raise FileNotFoundError(f"baseline weights not found: {weights} (run `make weights`)")
        state = torch.load(weights, map_location="cpu")
        try:
            self.net.load_state_dict(state)  # strict: a mismatch means the wrong file or repo commit
        except RuntimeError as e:
            raise RuntimeError(f"{weights} does not match the baseline RawNet architecture: {e}") from e

    # ---- shape facts ------------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """C_b: size of each feature vector in H_b (the GRU hidden size, 1024)."""
        return self.net.gru.hidden_size

    def feature_length(self, num_samples: int) -> int:
        """T_b for an input of `num_samples` samples (0 if the input is too short)."""
        n = num_samples - (self.net.Sinc_conv.kernel_size - 1)  # Sinc conv, no padding
        for _ in range(_NUM_POOL_STAGES):
            n //= _POOL_FACTOR
        return max(n, 0)

    @property
    def min_num_samples(self) -> int:
        """Shortest input that still leaves one time step."""
        return (self.net.Sinc_conv.kernel_size - 1) + _POOL_FACTOR ** _NUM_POOL_STAGES

    # ---- label / output conventions -----------------------------------------------------------
    @staticmethod
    def repo_label_to_baseline(label: torch.Tensor) -> torch.Tensor:
        """Repo binary label (0 bona fide / 1 synthetic) -> baseline target (1 bona fide / 0 synthetic)."""
        return 1 - label

    @staticmethod
    def baseline_label_to_repo(label: torch.Tensor) -> torch.Tensor:
        """Baseline binary target (1 bona fide / 0 synthetic) -> repo label (0 bona fide / 1 synthetic)."""
        return 1 - label

    @staticmethod
    def to_repo_binary(baseline_binary: torch.Tensor) -> torch.Tensor:
        """[B,2] baseline columns [synthetic, bona fide] -> repo columns [bona fide, synthetic]."""
        return baseline_binary.flip(1)

    # ---- pretrained heads ----------------------------------------------------------------------
    def pretrained_heads(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Copies of the baseline's head weights, ready for BinaryHead/VocoderHead.load_pretrained.

        'binary': fc2 rows are swapped so the class order is the repo's [bona fide, synthetic].
        'vocoder': unchanged. Keys: fc1.weight, fc1.bias, fc2.weight, fc2.bias.
        """
        n = self.net
        swap = [1, 0]
        return {
            "binary": {
                "fc1.weight": n.fc1_binary_gru.weight.detach().clone(),
                "fc1.bias": n.fc1_binary_gru.bias.detach().clone(),
                "fc2.weight": n.fc2_binary_gru.weight.detach()[swap].clone(),
                "fc2.bias": n.fc2_binary_gru.bias.detach()[swap].clone(),
            },
            "vocoder": {
                "fc1.weight": n.fc1_multi_gru.weight.detach().clone(),
                "fc1.bias": n.fc1_multi_gru.bias.detach().clone(),
                "fc2.weight": n.fc2_multi_gru.weight.detach().clone(),
                "fc2.bias": n.fc2_multi_gru.bias.detach().clone(),
            },
        }

    # ---- parameter groups / freezing -------------------------------------------------------------
    def head_parameters(self) -> Iterator[nn.Parameter]:
        for name in _HEAD_MODULES:
            yield from getattr(self.net, name).parameters()

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        """Everything in the baseline except its own four head linears."""
        for name, p in self.net.named_parameters():
            if not _is_head_param(name):
                yield p

    def set_freeze_encoder(self, freeze: bool) -> None:
        """Freeze / unfreeze the encoder (all parameters except the baseline's own heads)."""
        self.freeze_encoder = bool(freeze)
        for p in self.encoder_parameters():
            p.requires_grad_(not self.freeze_encoder)
        self.train(self.training)  # re-apply the eval-mode rule for the frozen part

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            for name, child in self.net.named_children():
                if name not in _HEAD_MODULES:
                    child.eval()  # frozen BatchNorm layers must not update their running statistics
        return self

    # ---- forward ----------------------------------------------------------------------------------
    def _check_wav(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() != 2:
            raise ValueError(f"expected a waveform batch [B, T], got shape {tuple(wav.shape)}")
        if not wav.is_floating_point():
            raise TypeError(f"expected a floating-point waveform, got {wav.dtype}")
        if wav.shape[1] < self.min_num_samples:
            raise ValueError(
                f"waveform has {wav.shape[1]} samples, the baseline needs at least {self.min_num_samples}"
            )
        return wav.contiguous()  # the baseline reshapes its input with .view()

    def _run(self, wav: torch.Tensor):
        wav = self._check_wav(wav)
        # SincConv builds its filters on the device it was constructed with; follow the input.
        self.net.Sinc_conv.device = wav.device
        self.net.device = wav.device
        captured = []
        handle = self.net.gru.register_forward_hook(lambda module, inputs, output: captured.append(output[0]))
        try:
            log_bin, log_voc = self.net(wav)
        finally:
            handle.remove()
        return log_bin, log_voc, captured[0]

    def forward_features(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [B,T] -> H_b [B, T_b, C_b]: the GRU output at every time step (before pooling)."""
        return self._run(wav)[2]

    def forward(self, wav: torch.Tensor) -> Dict[str, torch.Tensor]:
        log_bin, log_voc, h_b = self._run(wav)
        return {"logits_bin": self.to_repo_binary(log_bin), "logits_voc": log_voc, "H_b": h_b}

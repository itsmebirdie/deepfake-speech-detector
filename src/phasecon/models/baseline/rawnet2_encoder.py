"""Wrapper around the authors' released RawNet2 (+ vocoder-ID) model. Owner: Member C (step C1).

The baseline repo lives in third_party/ (git submodule). Do NOT copy its code into src/;
import it by adding `third_party/...` to sys.path inside this module only.
"""
from typing import Dict

import torch
import torch.nn as nn


class BaselineRawNet2(nn.Module):
    """Loads released weights and exposes the intermediate feature sequence.

    forward_features(wav [B,T]) -> H_b [B, T_b, C_b]   (temporal sequence BEFORE pooling)
    forward(wav) -> {'logits_bin': [B,2], 'logits_voc': [B,n_voc], 'H_b': H_b}   (parity with baseline)

    Responsibilities:
    - handle label-convention mapping (repo: 0 real / 1 fake) <-> baseline's convention
    - freeze_encoder flag for ablation (vii)
    - expose pretrained heads' weights so BinaryHead/VocoderHead can be initialised from them
    """

    def __init__(self, repo_dir: str, weights: str, freeze_encoder: bool, num_vocoder_classes: int):
        super().__init__()
        raise NotImplementedError("TODO(C1)")

    def forward_features(self, wav: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO(C1)")

    def forward(self, wav: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("TODO(C1)")

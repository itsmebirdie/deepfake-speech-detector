"""Full model. Owner: Member C (PR-18, param_groups for PR-20). Contract: docs/INTERFACES.md section 5."""
import warnings
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from phasecon.utils.logging import get_logger

from .baseline.rawnet2_encoder import BaselineRawNet2
from .dummy_phase_branch import DummyPhaseBranch
from .fusion import ConcatFusion, GatedCrossAttentionFusion
from .heads import BinaryHead, VocoderHead

logger = get_logger("phasecon.model")

_DEFAULT_DUMMY_T_D = 64


def _build_phase_branch(phase_cfg: Dict) -> Tuple[nn.Module, bool]:
    """Return (branch, is_dummy). `model.phase_branch.impl`: 'real' (default) | 'dummy'.

    'real' builds Member B's PhaseBranch (PR-21) and never falls back silently: if it cannot be built
    the error propagates. 'dummy' must be asked for explicitly; it returns the random-tensor
    DummyPhaseBranch, which is fine for plumbing tests and meaningless for results (scripts/train.py
    refuses to train on it unless told otherwise, and the warning below is emitted every time).
    """
    impl = phase_cfg.get("impl", "real")
    if impl not in ("real", "dummy"):
        raise ValueError(f"model.phase_branch.impl must be real or dummy, got {impl!r}")
    if impl == "real":
        from .phase_branch import PhaseBranch

        # PhaseBranch(cfg) receives the `model.phase_branch` sub-config (see INTERFACES.md section 3)
        return PhaseBranch(phase_cfg), False
    enc = phase_cfg.get("encoder", {}) or {}
    t_d = int(enc.get("t_out") or _DEFAULT_DUMMY_T_D)
    c_d = int(enc.get("out_dim", 128))
    msg = (
        "model.phase_branch.impl=dummy: using DummyPhaseBranch, which returns RANDOM features. "
        "Do not use this model for real training or results."
    )
    warnings.warn(msg, RuntimeWarning, stacklevel=3)
    logger.warning(msg)
    return DummyPhaseBranch(t_d=t_d, c_d=c_d), True


def _build_fusion(fusion_cfg: Dict, dim_b: int, dim_d: int) -> nn.Module:
    kind = fusion_cfg.get("type", "gated_cross_attention")
    if kind == "gated_cross_attention":
        return GatedCrossAttentionFusion(
            dim_b=dim_b,
            dim_d=dim_d,
            dim=int(fusion_cfg.get("dim", 256)),
            num_heads=int(fusion_cfg.get("num_heads", 4)),
            dropout=float(fusion_cfg.get("dropout", 0.1)),
            gate_bias_init=float(fusion_cfg.get("gate_bias_init", -3.0)),
        )
    if kind == "concat":
        return ConcatFusion(dim_b, dim_d)
    raise ValueError(f"model.fusion.type must be gated_cross_attention or concat, got {kind!r}")


class PhaseConNet(nn.Module):
    """forward(wav [B,T]) -> dict:
        'logits_bin':      [B,2]   raw logits, repo order [bona fide, synthetic]
        'logits_voc':      [B,n_voc]            (if vocoder head enabled)
        'cons_per_sample': [B]                  (from PhaseBranch; absent if phase branch disabled)
    With cfg model.phase_branch.enabled=false this reduces to the baseline: the heads start as copies
    of the baseline's pretrained heads and read the same last time step, so (log-)softmax of the
    outputs equals the baseline's outputs.

    Config read (all under cfg['model']): name (baseline_only | phasecon_net), baseline.{repo_dir,
    weights, freeze_encoder, num_vocoder_classes}, phase_branch.{enabled, impl (real | dummy, default
    real), encoder.{out_dim, t_out}, ...}, fusion.{type, dim, num_heads, dropout, gate_bias_init}, heads.{vocoder_head,
    gradient_reversal, grl_lambda, pooling (last | mean, default last)}.
    The phase stream is always run in float32, also under autocast (STFT / ULCP must not see fp16).
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        model_cfg = cfg["model"]
        name = model_cfg.get("name", "phasecon_net")
        base_cfg = model_cfg["baseline"]
        phase_cfg = model_cfg.get("phase_branch", {}) or {}
        heads_cfg = model_cfg.get("heads", {}) or {}
        enabled = bool(phase_cfg.get("enabled", False))
        if name == "baseline_only" and enabled:
            raise ValueError("model.name is baseline_only but model.phase_branch.enabled is true")
        self.pooling = heads_cfg.get("pooling", "last")
        if self.pooling not in ("last", "mean"):
            raise ValueError(f"model.heads.pooling must be last or mean, got {self.pooling!r}")
        use_voc = bool(heads_cfg.get("vocoder_head", True))
        grl = bool(heads_cfg.get("gradient_reversal", False))
        if grl and not use_voc:
            raise ValueError("heads.gradient_reversal needs heads.vocoder_head to be true")

        self.baseline = BaselineRawNet2(
            repo_dir=base_cfg["repo_dir"],
            weights=base_cfg.get("weights"),
            freeze_encoder=bool(base_cfg.get("freeze_encoder", False)),
            num_vocoder_classes=int(base_cfg.get("num_vocoder_classes", 7)),
        )
        c_b = self.baseline.feature_dim

        self.binary_head = BinaryHead(c_b)
        self.vocoder_head = (
            VocoderHead(c_b, self.baseline.num_vocoder_classes, gradient_reversal=grl,
                        lambd=float(heads_cfg.get("grl_lambda", 1.0)))
            if use_voc else None
        )
        pretrained = self.baseline.pretrained_heads()
        if not self.binary_head.load_pretrained(pretrained["binary"]):
            logger.warning("binary head not initialised from the baseline (shape mismatch)")
        if self.vocoder_head is not None and not self.vocoder_head.load_pretrained(pretrained["vocoder"]):
            logger.warning("vocoder head not initialised from the baseline (shape mismatch)")

        self.phase_enabled = enabled
        self.phase_branch = None
        self.fusion = None
        self.phase_branch_is_dummy = False
        if enabled:
            self.phase_branch, self.phase_branch_is_dummy = _build_phase_branch(phase_cfg)
            c_d = int((phase_cfg.get("encoder", {}) or {}).get("out_dim", 128))
            self.fusion = _build_fusion(model_cfg.get("fusion", {}) or {}, c_b, c_d)

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        return h[:, -1, :] if self.pooling == "last" else h.mean(dim=1)

    def forward(self, wav: torch.Tensor) -> Dict[str, torch.Tensor]:
        if wav.dim() != 2:
            raise ValueError(f"expected a waveform batch [B, T], got shape {tuple(wav.shape)}")
        h = self.baseline.forward_features(wav)  # [B, T_b, C_b]
        out: Dict[str, torch.Tensor] = {}
        if self.phase_branch is not None:
            with torch.autocast(device_type=wav.device.type, enabled=False):
                phase = self.phase_branch(wav.float())
            for key in ("H_d", "cons_per_sample"):
                if key not in phase:
                    raise KeyError(f"PhaseBranch output has no {key!r} (docs/INTERFACES.md section 3)")
            h = self.fusion(h, phase["H_d"])
            out["cons_per_sample"] = phase["cons_per_sample"]
        pooled = self._pool(h)
        out["logits_bin"] = self.binary_head(pooled)
        if self.vocoder_head is not None:
            out["logits_voc"] = self.vocoder_head(pooled)
        return out

    def param_groups(self, lr_encoder: float, lr_new: float, weight_decay: float) -> List[Dict]:
        """Separate LR for pretrained encoder vs new modules (used by training/optim.py).

        'encoder': the baseline's RawNet encoder (omitted when frozen). 'new': fusion, phase branch
        and the new heads. The baseline's own heads are not used by this model and are in no group.
        """
        baseline_ids = {id(p) for p in self.baseline.parameters()}
        encoder = [p for p in self.baseline.encoder_parameters() if p.requires_grad]
        new = [p for p in self.parameters() if p.requires_grad and id(p) not in baseline_ids]
        groups: List[Dict] = []
        if encoder:
            groups.append({"name": "encoder", "params": encoder, "lr": lr_encoder, "weight_decay": weight_decay})
        groups.append({"name": "new", "params": new, "lr": lr_new, "weight_decay": weight_decay})
        return groups

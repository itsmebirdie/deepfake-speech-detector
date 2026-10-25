"""Eq. 6: L = L_bin + beta * L_voc + mu * L_cons. Owner: Member C (PR-15)."""
from typing import Dict, Tuple

import torch

from .classification import binary_loss, vocoder_loss
from .consistency import consistency_loss  # Member B (PR-22); imported by name so tests can patch it


def total_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict,
    beta: float,
    mu: float,
    bin_class_weights: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Returns (loss, parts) where parts = {'bin':..., 'voc':..., 'cons':...} as floats for logging.

    Terms are skipped when they cannot or should not contribute:
    - 'voc' is absent from `parts` when `out` has no 'logits_voc' (no vocoder head);
    - 'cons' is absent when `out` has no 'cons_per_sample' (phase branch off);
    - a term with weight 0 (beta == 0 or mu == 0, ablation iv) is neither computed nor added, so it
      contributes no gradient and does not need Member B's code.
    `out['logits_bin']` and `batch['label']` are required; `batch['vocoder_id']` when 'voc' is used.
    """
    if "logits_bin" not in out:
        raise KeyError("model output has no 'logits_bin'")
    if beta < 0 or mu < 0:
        raise ValueError(f"beta and mu must be non-negative, got beta={beta}, mu={mu}")
    parts: Dict[str, float] = {}
    loss = binary_loss(out["logits_bin"], batch["label"], bin_class_weights)
    parts["bin"] = float(loss.detach())
    if "logits_voc" in out and beta != 0:
        if "vocoder_id" not in batch:
            raise KeyError("batch has no 'vocoder_id' but the model outputs 'logits_voc'")
        l_voc = vocoder_loss(out["logits_voc"], batch["vocoder_id"])
        parts["voc"] = float(l_voc.detach())
        loss = loss + beta * l_voc
    if "cons_per_sample" in out and mu != 0:
        l_cons = consistency_loss(out["cons_per_sample"], batch["label"])
        parts["cons"] = float(l_cons.detach())
        loss = loss + mu * l_cons
    return loss, parts

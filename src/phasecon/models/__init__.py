import warnings

from .baseline.rawnet2_encoder import NATIVE_NUM_SAMPLES, NATIVE_SAMPLE_RATE
from .phasecon_net import PhaseConNet  # noqa: F401

MODEL_NAMES = ("baseline_only", "phasecon_net")


def build_model(cfg):
    """Factory: cfg['model']['name'] in {'baseline_only', 'phasecon_net'}. Owner: Member C (PR-18).

    Both names build a PhaseConNet; 'baseline_only' requires model.phase_branch.enabled=false.
    """
    model_cfg = cfg.get("model")
    if not model_cfg or "name" not in model_cfg:
        raise KeyError("config has no model.name (expected one of %s)" % (MODEL_NAMES,))
    if model_cfg["name"] not in MODEL_NAMES:
        raise ValueError(f"unknown model.name {model_cfg['name']!r}; expected one of {MODEL_NAMES}")
    audio = cfg.get("audio", {})
    rate, samples = audio.get("sample_rate"), audio.get("num_samples")
    if (rate is not None and rate != NATIVE_SAMPLE_RATE) or (samples is not None and samples != NATIVE_NUM_SAMPLES):
        warnings.warn(
            f"audio config is {rate} Hz / {samples} samples but the baseline's Sinc front-end and released "
            f"weights assume {NATIVE_SAMPLE_RATE} Hz / {NATIVE_NUM_SAMPLES} samples "
            "(docs/DATASET_NOTES.md section 7, docs/INTERFACES.md section 4)",
            RuntimeWarning,
            stacklevel=2,
        )
    return PhaseConNet(cfg)

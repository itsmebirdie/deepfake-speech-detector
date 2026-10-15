from .phasecon_net import PhaseConNet  # noqa: F401


def build_model(cfg):
    """Factory: cfg['model']['name'] in {'baseline_only', 'phasecon_net'}. Owner: Member C (C4)."""
    raise NotImplementedError("TODO(C4)")

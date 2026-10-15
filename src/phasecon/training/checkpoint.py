"""Save/load checkpoints with config snapshot + git hash. Owner: Member C (step C6)."""
def save_checkpoint(path, model, optimizer, epoch, best_metric, cfg): raise NotImplementedError("TODO(C6)")
def load_checkpoint(path, model, optimizer=None, map_location="cpu"): raise NotImplementedError("TODO(C6)")

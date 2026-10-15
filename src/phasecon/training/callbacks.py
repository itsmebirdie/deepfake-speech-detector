"""Early stopping and metric logging (TensorBoard + CSV). Owner: Member C (step C6)."""
class EarlyStopping:
    def __init__(self, metric: str, mode: str, patience: int): raise NotImplementedError("TODO(C6)")
    def step(self, value: float) -> bool: raise NotImplementedError("TODO(C6)")
class MetricLogger:
    def __init__(self, out_dir: str): raise NotImplementedError("TODO(C6)")
    def log(self, step: int, scalars: dict): raise NotImplementedError("TODO(C6)")

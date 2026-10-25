"""Early stopping and metric logging (TensorBoard + CSV). Owner: Member C (PR-20)."""
import csv
import math
import os
from typing import Dict, List, Optional


class EarlyStopping:
    """Stops after `patience` consecutive epochs without improvement of `metric` (a key such as
    'dev_eer'). `step(value)` returns True when training should stop. NaN never counts as an
    improvement."""

    def __init__(self, metric: str, mode: str, patience: int, min_delta: float = 0.0):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        if patience < 1:
            raise ValueError(f"patience must be at least 1, got {patience}")
        self.metric, self.mode, self.patience, self.min_delta = metric, mode, patience, min_delta
        self.best: Optional[float] = None
        self.num_bad_epochs = 0
        self.is_best = False  # did the last step() improve the best value?

    def _improves(self, value: float) -> bool:
        if value is None or math.isnan(value):
            return False
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float) -> bool:
        self.is_best = self._improves(value)
        if self.is_best:
            self.best, self.num_bad_epochs = float(value), 0
        else:
            self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience

    def state_dict(self) -> Dict:
        return {"best": self.best, "num_bad_epochs": self.num_bad_epochs, "is_best": self.is_best}

    def load_state_dict(self, state: Dict) -> None:
        self.best, self.num_bad_epochs, self.is_best = state["best"], state["num_bad_epochs"], state["is_best"]


class MetricLogger:
    """Appends scalars to <out_dir>/metrics.csv (always) and to TensorBoard (if installed)."""

    def __init__(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, "metrics.csv")
        self._rows: List[Dict] = []
        self._tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._tb = SummaryWriter(os.path.join(out_dir, "tb"))
        except Exception:  # tensorboard is optional; the CSV is the record of truth
            self._tb = None

    def log(self, step: int, scalars: dict) -> None:
        self._rows.append({"step": int(step), **scalars})
        columns: List[str] = []
        for row in self._rows:  # header = every key, in order of first appearance
            columns.extend(k for k in row if k not in columns)
        tmp = self.csv_path + ".tmp"
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, restval="")
            writer.writeheader()
            writer.writerows(self._rows)
        os.replace(tmp, self.csv_path)
        if self._tb is not None:
            for key, value in scalars.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    self._tb.add_scalar(key, value, step)
            self._tb.flush()

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()

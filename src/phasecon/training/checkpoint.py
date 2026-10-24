"""Save/load checkpoints with config snapshot + git hash. Owner: Member C (PR-20)."""
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]


def git_info(cwd: Optional[str] = None) -> Dict:
    """{'git_hash': commit or 'unknown', 'git_dirty': bool} for the repo this code lives in."""
    cwd = cwd or str(_REPO_ROOT)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip())
        return {"git_hash": commit, "git_dirty": dirty}
    except Exception:
        return {"git_hash": "unknown", "git_dirty": False}


def save_checkpoint(path, model, optimizer, epoch, best_metric, cfg, scheduler=None, scaler=None, extra=None):
    """Write model + optimizer (+ scheduler / AMP scaler) state, epoch, best metric, a snapshot of
    the config and the git commit. Atomic: a crash mid-write leaves any older file intact."""
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "best_metric": None if best_metric is None else float(best_metric),
        "cfg": cfg,
        "torch_version": torch.__version__,
        "extra": extra or {},
        **git_info(),
    }
    path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu") -> Dict:
    """Restore model (strict) and, if given, optimizer state. Returns the full checkpoint dict
    (epoch, best_metric, cfg, git_hash, scheduler/scaler state, ...). Only load files you trust:
    checkpoints hold a pickled config."""
    ckpt = torch.load(str(path), map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        if ckpt.get("optimizer") is None:
            raise ValueError(f"{path} has no optimizer state")
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt

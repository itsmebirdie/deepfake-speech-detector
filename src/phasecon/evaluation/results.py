"""Results registry: one standard on-disk format for every run (PR-17). Owner: Member A.

    experiments/results/<run_name>/
        config.yaml    the full merged config the run used
        metrics.json   {eer, eer_threshold, auc, accuracy, per_vocoder, ...} + git_hash, git_dirty, created
        scores.csv     one row per clip: utt_id, label, vocoder_id, score   (higher = more synthetic)

Contract: docs/INTERFACES.md section 6. Later aggregation (PR-31) only ever reads this format.
"""
import csv
import json
import os
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import yaml

from phasecon.evaluation.evaluate import save_scores
from phasecon.training.checkpoint import git_info

REQUIRED_METRICS = ("eer", "eer_threshold", "auc", "accuracy")


def write_results(run_dir: str, cfg: Dict, metrics: Dict, scores: Optional[Dict] = None,
                  overwrite: bool = False) -> str:
    """Write config.yaml, metrics.json (+ git hash) and, if given, scores.csv into `run_dir`.

    `scores` is the dict returned by `evaluate.run_inference`. Refuses to touch a folder that
    already holds results unless `overwrite=True`, so a finished run is never silently replaced.
    """
    missing = [k for k in REQUIRED_METRICS if k not in metrics]
    if missing:
        raise ValueError(f"metrics is missing {missing}")
    if os.path.exists(os.path.join(run_dir, "metrics.json")) and not overwrite:
        raise FileExistsError(f"{run_dir} already has results; pick another run name or pass overwrite=True")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    record = dict(metrics)
    record.update(git_info())
    record["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(record, f, indent=2)
    if scores is not None:
        save_scores(scores, os.path.join(run_dir, "scores.csv"))
    return run_dir


def read_results(run_dir: str) -> Dict:
    """Read a results folder back: {'config', 'metrics', 'scores' (arrays, or None if no scores.csv)}."""
    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(run_dir, "metrics.json")) as f:
        metrics = json.load(f)
    scores = None
    path = os.path.join(run_dir, "scores.csv")
    if os.path.exists(path):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        scores = {
            "utt_id": [r["utt_id"] for r in rows],
            "label": np.array([int(r["label"]) for r in rows]),
            "vocoder_id": np.array([int(r["vocoder_id"]) for r in rows]),
            "score": np.array([float(r["score"]) for r in rows]),
        }
    return {"config": cfg, "metrics": metrics, "scores": scores}

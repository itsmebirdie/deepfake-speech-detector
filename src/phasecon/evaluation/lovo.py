"""Leave-one-vocoder-out driver. Owner: Member A (step A7 for baseline, A9 for PhaseCon-Net).

For each held-out vocoder V (protocol: docs/DATASET_NOTES.md section 10):
    1. train a FRESH model on the fold's train/dev split (bona fide + the 5 other vocoders)
    2. evaluate its best checkpoint on the fold's test split (bona fide + only V's fakes)
    3. save the result through the results registry as <prefix>_lovo-<V>
Then `summarize_lovo` reports every fold and the mean over the six folds (the number the report uses), plus a
pooled EER over all folds' scores for reference.

Folds whose results folder already exists are skipped, so an interrupted run resumes where it stopped.
The training and evaluation steps can be passed in (`train_fold`, `eval_fold`); the defaults run
`training.Trainer` and the same inference code as `scripts/evaluate.py`, in this process.
"""
import copy
import os
from typing import Callable, Dict, List, Optional

import numpy as np

from phasecon.evaluation.metrics import compute_eer
from phasecon.evaluation.results import read_results

TrainFn = Callable[[Dict], str]          # fold cfg -> path of the best checkpoint
EvalFn = Callable[[Dict, str, str], Dict]  # (fold cfg, checkpoint, results dir) -> metrics


def fold_run_name(prefix: str, held_out: str) -> str:
    return f"{prefix}_lovo-{held_out}"


def fold_config(cfg: Dict, held_out: str, prefix: str) -> Dict:
    """A copy of cfg for one fold: the fold's splits, LOVO evaluation, and its own run name."""
    vocoders = cfg["data"]["vocoders"]
    if held_out not in vocoders:
        raise ValueError(f"held_out={held_out!r} is not one of {vocoders}")
    fold = copy.deepcopy(cfg)
    fold["data"].setdefault("lovo", {})["held_out"] = held_out
    fold.setdefault("eval", {})["protocol"] = "leave_one_vocoder_out"
    fold["run_name"] = f"{fold_run_name(prefix, held_out)}_train"
    return fold


def _default_train(fold_cfg: Dict) -> str:
    from phasecon.data.librisevoc import build_dataloader
    from phasecon.models import build_model
    from phasecon.training.trainer import Trainer
    from phasecon.utils.seed import set_seed

    set_seed(int(fold_cfg.get("seed", 0)), bool(fold_cfg.get("deterministic", False)))
    model = build_model(fold_cfg)
    train_loader = build_dataloader(fold_cfg, "train", train=True)
    dev_loader = build_dataloader(fold_cfg, "dev", train=False)
    return Trainer(fold_cfg, model, train_loader, dev_loader).fit()


def _default_eval(fold_cfg: Dict, checkpoint: str, run_dir: str) -> Dict:
    import torch

    from phasecon.data.librisevoc import build_dataloader
    from phasecon.evaluation.evaluate import run_inference
    from phasecon.evaluation.metrics import summarize
    from phasecon.evaluation.results import write_results
    from phasecon.models import build_model
    from phasecon.training.checkpoint import load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_cfg = copy.deepcopy(fold_cfg)
    eval_cfg["model"].get("baseline", {})["weights"] = None  # everything comes from the checkpoint
    model = build_model(eval_cfg).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    loader = build_dataloader(eval_cfg, "test", train=False)
    scores = run_inference(model, loader, device)
    metrics = summarize(scores["score"], scores["label"], scores["vocoder_id"], eval_cfg["data"]["vocoders"])
    metrics.update({"run_name": os.path.basename(run_dir), "split": "test",
                    "protocol": "leave_one_vocoder_out", "held_out": eval_cfg["data"]["lovo"]["held_out"],
                    "checkpoint": checkpoint, "device": device})
    write_results(run_dir, eval_cfg, metrics, scores)
    return metrics


def run_lovo(cfg: Dict, model_name: str, vocoders: List[str], prefix: Optional[str] = None,
             train_fold: Optional[TrainFn] = None, eval_fold: Optional[EvalFn] = None) -> Dict:
    """For each held-out vocoder: build split -> train fresh model (baseline arch or PhaseCon-Net) -> evaluate.
    Returns {vocoder: metrics}. Baseline must be RETRAINED per fold (released weights saw all vocoders)."""
    if cfg["model"]["name"] != model_name:
        raise ValueError(f"cfg is for model {cfg['model']['name']!r}, not {model_name!r}")
    train_fold = train_fold or _default_train
    eval_fold = eval_fold or _default_eval
    prefix = prefix or f"{model_name}_lovo"
    results_root = os.path.join(cfg.get("output_dir", "experiments"), "results")
    out: Dict[str, Dict] = {}
    for held_out in vocoders:
        run_dir = os.path.join(results_root, fold_run_name(prefix, held_out))
        if os.path.exists(os.path.join(run_dir, "metrics.json")):
            print(f"[lovo] {held_out}: results exist, skipping ({run_dir})", flush=True)
            out[held_out] = read_results(run_dir)["metrics"]
            continue
        fold = fold_config(cfg, held_out, prefix)
        print(f"[lovo] {held_out}: training {fold['run_name']}", flush=True)
        checkpoint = train_fold(fold)
        print(f"[lovo] {held_out}: evaluating {checkpoint}", flush=True)
        out[held_out] = eval_fold(fold, checkpoint, run_dir)
        print(f"[lovo] {held_out}: EER {100 * out[held_out]['eer']:.3f}%", flush=True)
    return out


def summarize_lovo(per_fold: Dict[str, Dict], results_root: Optional[str] = None,
                   prefix: Optional[str] = None) -> Dict:
    """Per-fold EER/AUC/accuracy, their mean over folds, and (if scores are saved) a pooled EER."""
    keys = ("eer", "auc", "accuracy")
    summary = {
        "folds": {v: {k: float(m[k]) for k in keys} for v, m in per_fold.items()},
        "mean": {k: float(np.mean([m[k] for m in per_fold.values()])) for k in keys},
        "num_folds": len(per_fold),
    }
    if results_root and prefix:
        scores, labels = [], []
        for v in per_fold:
            s = read_results(os.path.join(results_root, fold_run_name(prefix, v)))["scores"]
            if s is None:
                break
            scores.append(np.asarray(s["score"], dtype=float))
            labels.append(np.asarray(s["label"], dtype=int))
        else:
            if scores:
                summary["pooled_eer"] = float(compute_eer(np.concatenate(scores), np.concatenate(labels))[0])
    return summary


def format_lovo_table(summary: Dict) -> str:
    """Markdown table: one row per held-out vocoder, then the mean (and pooled EER if present)."""
    lines = ["| Held-out vocoder | EER | AUC | Accuracy |", "|---|---|---|---|"]
    for v, m in summary["folds"].items():
        lines.append(f"| {v} | {100 * m['eer']:.2f}% | {m['auc']:.4f} | {100 * m['accuracy']:.2f}% |")
    m = summary["mean"]
    lines.append(f"| **mean of {summary['num_folds']} folds** | **{100 * m['eer']:.2f}%** | {m['auc']:.4f} | "
                 f"{100 * m['accuracy']:.2f}% |")
    if "pooled_eer" in summary:
        lines.append(f"\nPooled EER over all folds' scores (different models per fold; for reference only): "
                     f"{100 * summary['pooled_eer']:.2f}%")
    return "\n".join(lines) + "\n"

"""Evaluate a model on a split and save a standard results folder (PR-17). Owner: A5.

Loads configs/default.yaml + configs/data/librisevoc.yaml, then `--config` files on top
(default: the baseline-only model + configs/eval/in_distribution.yaml), then `--override`.
Builds the model with `build_model`, optionally loads a checkpoint from `scripts/train.py`, scores
every clip of `eval.split` and writes experiments/results/<run_name>/{config.yaml, metrics.json, scores.csv}.

Examples (from the repo root):
  # released baseline, official test split (it needs 24 kHz / 96,000 samples)
  python scripts/evaluate.py --override model.baseline.repo_dir=../baseline-repo audio.sample_rate=24000 audio.num_samples=96000

  # a trained checkpoint on one LOVO fold
  python scripts/evaluate.py --config configs/model/baseline_only.yaml configs/eval/lovo.yaml \\
      --checkpoint experiments/checkpoints/<run>/best.pt --override data.lovo.held_out=wavernn
"""
import argparse
import os
import time
from pathlib import Path

import torch

from phasecon.data.librisevoc import build_dataloader
from phasecon.evaluation.evaluate import run_inference
from phasecon.evaluation.metrics import summarize
from phasecon.evaluation.results import write_results
from phasecon.models import build_model
from phasecon.training.checkpoint import load_checkpoint
from phasecon.utils.config import load_many

REPO = Path(__file__).resolve().parents[1]
BASE_CONFIGS = ["configs/default.yaml", "configs/data/librisevoc.yaml"]
DEFAULT_CONFIGS = ["configs/model/baseline_only.yaml", "configs/eval/in_distribution.yaml"]


def default_run_name(cfg, checkpoint, split, held_out):
    source = "released" if checkpoint is None else Path(checkpoint).parent.name or Path(checkpoint).stem
    name = f"{cfg['model']['name']}_{source}_{split}"
    return f"{name}_lovo-{held_out}" if held_out else name


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate a model (in-distribution or LOVO) and save results.")
    ap.add_argument("--config", nargs="*", default=DEFAULT_CONFIGS,
                    help="YAML configs merged on top of the default/data configs, left to right")
    ap.add_argument("--override", nargs="*", default=[], help="dotted overrides, e.g. eval.split=dev")
    ap.add_argument("--checkpoint", default=None, help="checkpoint from scripts/train.py (default: none)")
    ap.add_argument("--run-name", default=None, help="results folder name (default: built from model/split)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing results folder")
    args = ap.parse_args(argv)

    def rel(p):  # config paths are relative to the repo root unless they exist as given
        return p if Path(p).exists() else str(REPO / p)

    cfg = load_many([rel(p) for p in BASE_CONFIGS + list(args.config)], args.override)
    eval_cfg = cfg.get("eval", {})
    split = eval_cfg.get("split", "test")
    held_out = cfg["data"].get("lovo", {}).get("held_out")
    if eval_cfg.get("protocol") == "leave_one_vocoder_out" and not held_out:
        raise SystemExit("LOVO evaluation needs a fold: add --override data.lovo.held_out=<vocoder>")
    run_name = args.run_name or default_run_name(cfg, args.checkpoint, split, held_out)
    run_dir = os.path.join(cfg.get("output_dir", "experiments"), "results", run_name)
    if os.path.exists(os.path.join(run_dir, "metrics.json")) and not args.overwrite:
        raise SystemExit(f"{run_dir} already has results; use --run-name or --overwrite")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location=device)
    loader = build_dataloader(cfg, split, train=False)
    print(f"evaluating {run_name}: split={split} clips={len(loader.dataset)} device={device} "
          f"checkpoint={args.checkpoint or 'none (released weights)'}", flush=True)

    start = time.time()
    scores = run_inference(model, loader, device)
    metrics = summarize(scores["score"], scores["label"], scores["vocoder_id"], cfg["data"]["vocoders"])
    metrics.update({
        "run_name": run_name,
        "split": split,
        "protocol": eval_cfg.get("protocol", "in_distribution"),
        "held_out": held_out,
        "checkpoint": args.checkpoint,
        "device": device,
        "minutes": round((time.time() - start) / 60, 1),
    })
    write_results(run_dir, cfg, metrics, scores if eval_cfg.get("save_scores", True) else None,
                  overwrite=args.overwrite)

    print(f"DONE -> {run_dir}")
    print(f"EER {100 * metrics['eer']:.3f}%  AUC {metrics['auc']:.5f}  accuracy {100 * metrics['accuracy']:.2f}%")
    for name, v in metrics.get("per_vocoder", {}).items():
        print(f"  {name:18s} EER {100 * v['eer']:.3f}%  caught {100 * v['caught']:.2f}%")
    return run_dir


if __name__ == "__main__":
    main()

"""Train PhaseCon-Net (or baseline-only) from config.  Owner: Member C (PR-20).

Examples (from the repo root):
  python scripts/train.py --config configs/model/baseline_only.yaml --override train.epochs=1
  python scripts/train.py --config configs/ablation/a3_concat_fusion.yaml run_name=a3_seed1
  python scripts/train.py --overfit-batch --override train.overfit_steps=200   # sanity check

configs/default.yaml, configs/data/librisevoc.yaml and configs/train/default.yaml are always loaded
first; `--config` files are merged on top, left to right (default: configs/model/phasecon.yaml);
`--override` applies dotted `a.b.c=value` edits last.
"""
import argparse
import sys
from pathlib import Path

import yaml

from phasecon.data.librisevoc import build_dataloader
from phasecon.models import build_model
from phasecon.training.trainer import Trainer
from phasecon.utils.config import load_many
from phasecon.utils.logging import get_logger
from phasecon.utils.seed import set_seed

REPO = Path(__file__).resolve().parents[1]
BASE_CONFIGS = ["configs/default.yaml", "configs/data/librisevoc.yaml", "configs/train/default.yaml"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train PhaseCon-Net (or baseline-only) from config.")
    ap.add_argument("--config", nargs="*", default=["configs/model/phasecon.yaml"],
                    help="YAML configs merged on top of the default/data/train configs, left to right")
    ap.add_argument("--override", nargs="*", default=[], help="dotted overrides, e.g. train.epochs=1")
    ap.add_argument("--overfit-batch", action="store_true", help="train on one batch only (sanity check)")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--allow-dummy-phase-branch", action="store_true",
                    help="allow the random-feature DummyPhaseBranch (plumbing tests only, never for results)")
    args = ap.parse_args(argv)
    log = get_logger("phasecon.train_script")

    def rel(p):  # config paths are relative to the repo root unless they exist as given
        return p if Path(p).exists() else str(REPO / p)

    overrides = list(args.override)
    if args.overfit_batch:
        overrides.append("train.overfit_batch=true")
    if args.resume:
        overrides.append(f"train.resume={args.resume}")
    cfg = load_many([rel(p) for p in BASE_CONFIGS + list(args.config)], overrides)

    set_seed(int(cfg.get("seed", 0)), bool(cfg.get("deterministic", False)))
    model = build_model(cfg)
    if getattr(model, "phase_branch_is_dummy", False) and not args.allow_dummy_phase_branch:
        sys.exit("refusing to train: model.phase_branch.impl=dummy selects the random-feature DummyPhaseBranch. "
                 "Remove that override (the default is the real PhaseBranch), or pass --allow-dummy-phase-branch "
                 "for plumbing tests only.")
    overfit = bool(cfg["train"].get("overfit_batch"))
    train_loader = build_dataloader(cfg, "train", train=True)
    dev_loader = None if overfit else build_dataloader(cfg, "dev", train=False)
    log.info("config:\n%s", yaml.safe_dump(cfg, sort_keys=False))
    trainer = Trainer(cfg, model, train_loader, dev_loader)
    best = trainer.fit()
    print(f"DONE: best checkpoint -> {best}")
    return best


if __name__ == "__main__":
    main()

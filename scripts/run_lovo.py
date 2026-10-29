"""PR-26: retrain the baseline once per held-out vocoder and evaluate each fold. Owner: Member A.

Same recipe as PR-24's fine-tuned baseline (scripts/run_pr24.sh, DATASET_NOTES section 11) so the numbers are
comparable, with one deliberate difference: training starts from RANDOM weights, not the released ones, because
the released weights were trained on every vocoder (a "held-out" vocoder would not really be unseen).

  python scripts/run_lovo.py                                   # all six folds, then the summary
  python scripts/run_lovo.py --vocoders wavernn                # one fold
  python scripts/run_lovo.py --override train.epochs=1 data.subset.per_vocoder_train=20 \\
      data.subset.per_vocoder_dev=10 --prefix smoke            # tiny run to test the pipeline on a laptop
  python scripts/run_lovo.py --summary-only                    # rebuild the table from finished folds

Finished folds are skipped, so re-running resumes. Results: experiments/results/<prefix>_lovo-<vocoder>/ and
experiments/results/<prefix>_summary.{json,md}.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from phasecon.evaluation.lovo import format_lovo_table, fold_run_name, run_lovo, summarize_lovo
from phasecon.evaluation.results import read_results
from phasecon.utils.config import load_many

REPO = Path(__file__).resolve().parents[1]
BASE_CONFIGS = ["configs/default.yaml", "configs/data/librisevoc.yaml", "configs/train/default.yaml"]
# PR-24 fine-tuning recipe (scripts/run_pr24.sh), plus training from scratch
PR24_RECIPE = [
    "data.subset.enabled=true", "train.loss.beta=1.0", "train.loss.mu=0.0",
    "train.optimizer.weight_decay=2e-4", "train.optimizer.lr_encoder=1e-4", "train.optimizer.lr_new=1e-4",
    "model.baseline.freeze_encoder=false", "model.baseline.weights=null",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Baseline leave-one-vocoder-out retraining (PR-26).")
    ap.add_argument("--config", nargs="*", default=["configs/model/baseline_only.yaml", "configs/eval/lovo.yaml"])
    ap.add_argument("--vocoders", nargs="*", default=None, help="folds to run (default: all six)")
    ap.add_argument("--prefix", default="baseline_lovo", help="results are <prefix>_lovo-<vocoder>")
    ap.add_argument("--override", nargs="*", default=[], help="extra dotted overrides, applied last")
    ap.add_argument("--summary-only", action="store_true", help="only rebuild the summary from finished folds")
    args = ap.parse_args(argv)

    def rel(p):
        return p if Path(p).exists() else str(REPO / p)

    cfg = load_many([rel(p) for p in BASE_CONFIGS + list(args.config)], PR24_RECIPE + list(args.override))
    vocoders = args.vocoders or list(cfg["data"]["vocoders"])
    results_root = os.path.join(cfg.get("output_dir", "experiments"), "results")

    if args.summary_only:
        per_fold = {}
        for v in vocoders:
            run_dir = os.path.join(results_root, fold_run_name(args.prefix, v))
            if os.path.exists(os.path.join(run_dir, "metrics.json")):
                per_fold[v] = read_results(run_dir)["metrics"]
        if not per_fold:
            sys.exit(f"no finished folds for prefix {args.prefix!r} in {results_root}")
    else:
        per_fold = run_lovo(cfg, cfg["model"]["name"], vocoders, prefix=args.prefix)

    summary = summarize_lovo(per_fold, results_root, args.prefix)
    table = format_lovo_table(summary)
    os.makedirs(results_root, exist_ok=True)
    with open(os.path.join(results_root, f"{args.prefix}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(results_root, f"{args.prefix}_summary.md"), "w") as f:
        f.write(table)
    print("\n" + table)
    missing = [v for v in cfg["data"]["vocoders"] if v not in per_fold]
    if missing:
        print(f"Folds not finished yet: {', '.join(missing)}")
    return summary


if __name__ == "__main__":
    main()

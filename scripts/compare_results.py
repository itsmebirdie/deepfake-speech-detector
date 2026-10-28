"""Compare evaluation results against a reference and check tolerances (PR-24). Owner: Member C.

Reads the `metrics.json` of any run folder written by `scripts/evaluate.py` (PR-17) or by
`scripts/eval_baseline.py` (PR-06); the two use different key names and this script normalises both.

Reference: `--reference pr06` (default) = the released weights as measured in PR-06, copied from
docs/DATASET_NOTES.md section 8, or the path of any results folder / metrics.json.

Two checks, both optional:
  --parity       the run must MATCH the reference (same weights through a different code path, so
                 only floating-point noise is allowed). Defaults: EER 0.05 pp, AUC 0.001,
                 accuracy 0.1 pp, each vocoder's EER 0.1 pp.
  --no-regress   a TRAINED run must not be WORSE than the reference: pooled EER at most
                 `--max-regression` (default 0.5 pp) above it. Getting better is expected and fine.
Exit code 1 if a requested check fails, so it can stop a script.

Examples:
  python scripts/compare_results.py experiments/results/baseline_pipeline_released --parity
  python scripts/compare_results.py experiments/results/baseline_pipeline_frozen \\
      experiments/results/baseline_pipeline_finetuned --no-regress
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

# Released weights on the official test split (docs/DATASET_NOTES.md section 8, PR-06).
PR06 = {
    "eer": 0.1186,
    "auc": 0.9438,
    "accuracy": 0.7537,
    "per_vocoder": {
        "wavegrad": 0.0011, "diffwave": 0.0019, "parallel_wave_gan": 0.0017,
        "wavernn": 0.3654, "wavenet": 0.0814, "melgan": 0.0051,
    },
}
PARITY_TOLERANCE = {"eer": 0.0005, "auc": 0.001, "accuracy": 0.001, "per_vocoder": 0.001}
DEFAULT_MAX_REGRESSION = 0.005


def _metrics_path(path: str) -> str:
    return os.path.join(path, "metrics.json") if os.path.isdir(path) else path


def normalise(raw: Dict) -> Dict:
    """Map either metrics.json layout to {eer, auc, accuracy, per_vocoder{name: eer}}, all fractions."""
    eer = raw["eer"] if "eer" in raw else raw["eer_percent"] / 100.0
    accuracy = raw["accuracy"] if "accuracy" in raw else raw["accuracy_argmax"]
    per_vocoder = {}
    for name, v in (raw.get("per_vocoder") or {}).items():
        per_vocoder[name] = v["eer"] if "eer" in v else v["eer_percent"] / 100.0
    return {"eer": float(eer), "auc": float(raw["auc"]), "accuracy": float(accuracy), "per_vocoder": per_vocoder}


def load_metrics(path: str) -> Dict:
    with open(_metrics_path(path)) as f:
        return normalise(json.load(f))


def load_reference(spec: str) -> Dict:
    return PR06 if spec == "pr06" else load_metrics(spec)


def parity_checks(run: Dict, ref: Dict, tol: Optional[Dict] = None) -> List[Dict]:
    """One row per metric: {name, run, ref, delta, tolerance, ok}. Vocoders missing from either side are skipped."""
    tol = {**PARITY_TOLERANCE, **(tol or {})}
    rows = []
    for key in ("eer", "auc", "accuracy"):
        rows.append(_row(key, run[key], ref[key], tol[key]))
    for name in ref["per_vocoder"]:
        if name in run["per_vocoder"]:
            rows.append(_row(f"EER {name}", run["per_vocoder"][name], ref["per_vocoder"][name], tol["per_vocoder"]))
    return rows


def regression_check(run: Dict, ref: Dict, max_regression: float = DEFAULT_MAX_REGRESSION) -> Dict:
    """Pooled EER may not be more than `max_regression` above the reference (lower is better)."""
    row = _row("eer", run["eer"], ref["eer"], max_regression)
    row["ok"] = run["eer"] - ref["eer"] <= max_regression
    return row


def _row(name, value, ref, tolerance) -> Dict:
    delta = value - ref
    return {"name": name, "run": value, "ref": ref, "delta": delta, "tolerance": tolerance, "ok": abs(delta) <= tolerance}


def format_table(entries: List[tuple]) -> str:
    """Markdown table: entries = [(label, normalised metrics), ...]; the first is the reference."""
    vocoders: List[str] = []
    for _, m in entries:
        vocoders.extend(v for v in m["per_vocoder"] if v not in vocoders)
    head = ["Run", "EER %", "AUC", "Acc %"] + [f"{v} EER %" for v in vocoders]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for label, m in entries:
        cells = [label, f"{100 * m['eer']:.2f}", f"{m['auc']:.4f}", f"{100 * m['accuracy']:.2f}"]
        cells += [f"{100 * m['per_vocoder'][v]:.2f}" if v in m["per_vocoder"] else "" for v in vocoders]
        lines.append("| " + " | ".join(cells) + " |")
    ref = entries[0][1]
    if len(entries) > 1:
        lines += ["", "Change vs the reference, in percentage points for EER and accuracy (negative EER = better):", ""]
        lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for label, m in entries[1:]:
            cells = [label, f"{100 * (m['eer'] - ref['eer']):+.2f}", f"{m['auc'] - ref['auc']:+.4f}",
                     f"{100 * (m['accuracy'] - ref['accuracy']):+.2f}"]
            for v in vocoders:
                both = v in m["per_vocoder"] and v in ref["per_vocoder"]
                cells.append(f"{100 * (m['per_vocoder'][v] - ref['per_vocoder'][v]):+.2f}" if both else "")
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare results folders against a reference (PR-24).")
    ap.add_argument("runs", nargs="+", help="results folders (or metrics.json files) to compare")
    ap.add_argument("--reference", default="pr06", help="'pr06' (default) or a results folder / metrics.json")
    ap.add_argument("--parity", action="store_true", help="require each run to match the reference")
    ap.add_argument("--no-regress", action="store_true", help="require each run to be no worse than the reference")
    ap.add_argument("--max-regression", type=float, default=DEFAULT_MAX_REGRESSION,
                    help="allowed pooled-EER increase for --no-regress, as a fraction (0.005 = 0.5 pp)")
    ap.add_argument("--tol-eer", type=float, default=None, help="parity tolerance on pooled EER (fraction)")
    ap.add_argument("--tol-auc", type=float, default=None)
    ap.add_argument("--tol-accuracy", type=float, default=None)
    ap.add_argument("--tol-vocoder", type=float, default=None, help="parity tolerance on each vocoder's EER")
    ap.add_argument("--out", default=None, help="also write the markdown table here")
    args = ap.parse_args(argv)

    ref = load_reference(args.reference)
    entries = [(f"reference ({args.reference})", ref)]
    runs = []
    for path in args.runs:
        m = load_metrics(path)
        runs.append((path, m))
        entries.append((os.path.basename(os.path.normpath(path)), m))
    table = format_table(entries)
    print(table)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")

    tol = {k: v for k, v in (("eer", args.tol_eer), ("auc", args.tol_auc), ("accuracy", args.tol_accuracy),
                             ("per_vocoder", args.tol_vocoder)) if v is not None}
    failed = False
    for path, m in runs:
        if args.parity:
            print(f"\nparity check: {path}")
            for r in parity_checks(m, ref, tol):
                scale = 1.0 if r["name"] == "auc" else 100.0
                unit = "" if r["name"] == "auc" else " pp"
                print(f"  {'PASS' if r['ok'] else 'FAIL'}  {r['name']:22s} run {scale * r['run']:.4f}  "
                      f"ref {scale * r['ref']:.4f}  diff {scale * r['delta']:+.4f}{unit}  "
                      f"(allowed {scale * r['tolerance']:.4f}{unit})")
                failed |= not r["ok"]
        if args.no_regress:
            r = regression_check(m, ref, args.max_regression)
            print(f"\nno-regression check: {path}\n  {'PASS' if r['ok'] else 'FAIL'}  pooled EER {100 * r['run']:.2f}%  "
                  f"vs reference {100 * r['ref']:.2f}%  ({100 * r['delta']:+.2f} pp, allowed up to "
                  f"{100 * r['tolerance']:+.2f} pp)")
            failed |= not r["ok"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

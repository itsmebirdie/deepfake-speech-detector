"""Launch ablation configs a1..a7 (incl. K / R sweeps) and collect results.  Owner: A9."""
import argparse


def main():
    ap = argparse.ArgumentParser(description="Launch ablation configs a1..a7 (incl. K / R sweeps) and collect results.")
    ap.add_argument("--config", nargs="+", required=False, help="one or more YAML configs (merged left to right)")
    ap.add_argument("--override", nargs="*", default=[], help="dotted overrides, e.g. train.epochs=1")
    args = ap.parse_args()
    raise NotImplementedError("TODO(A9)")


if __name__ == "__main__":
    main()

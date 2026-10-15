"""Evaluate a checkpoint (in-distribution or LOVO).  Owner: A5."""
import argparse


def main():
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint (in-distribution or LOVO).")
    ap.add_argument("--config", nargs="+", required=False, help="one or more YAML configs (merged left to right)")
    ap.add_argument("--override", nargs="*", default=[], help="dotted overrides, e.g. train.epochs=1")
    ap.parse_args()
    raise NotImplementedError("TODO(A5)")


if __name__ == "__main__":
    main()

"""Evaluate released pretrained baseline on the official test split.  Owner: A4."""
import argparse


def main():
    ap = argparse.ArgumentParser(description="Evaluate released pretrained baseline on the official test split.")
    ap.add_argument("--config", nargs="+", required=False, help="one or more YAML configs (merged left to right)")
    ap.add_argument("--override", nargs="*", default=[], help="dotted overrides, e.g. train.epochs=1")
    ap.parse_args()
    raise NotImplementedError("TODO(A4)")


if __name__ == "__main__":
    main()

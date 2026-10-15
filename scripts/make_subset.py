"""Build fixed class-balanced subset manifests.  Owner: A6."""
import argparse


def main():
    ap = argparse.ArgumentParser(description="Build fixed class-balanced subset manifests.")
    ap.add_argument("--config", nargs="+", required=False, help="one or more YAML configs (merged left to right)")
    ap.add_argument("--override", nargs="*", default=[], help="dotted overrides, e.g. train.epochs=1")
    ap.parse_args()
    raise NotImplementedError("TODO(A6)")


if __name__ == "__main__":
    main()

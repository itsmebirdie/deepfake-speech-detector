"""Train PhaseCon-Net (or baseline-only) from config.  Owner: C6."""
import argparse


def main():
    ap = argparse.ArgumentParser(description="Train PhaseCon-Net (or baseline-only) from config.")
    ap.add_argument("--config", nargs="+", required=False, help="one or more YAML configs (merged left to right)")
    ap.add_argument("--override", nargs="*", default=[], help="dotted overrides, e.g. train.epochs=1")
    ap.parse_args()
    raise NotImplementedError("TODO(C6)")


if __name__ == "__main__":
    main()

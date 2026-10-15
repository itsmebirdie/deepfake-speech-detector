"""Plot delta_phi maps for real vs fake clips (debug / report figures).  Owner: B3."""
import argparse


def main():
    ap = argparse.ArgumentParser(description="Plot delta_phi maps for real vs fake clips (debug / report figures).")
    ap.add_argument("--config", nargs="+", required=False, help="one or more YAML configs (merged left to right)")
    ap.add_argument("--override", nargs="*", default=[], help="dotted overrides, e.g. train.epochs=1")
    ap.parse_args()
    raise NotImplementedError("TODO(B3)")


if __name__ == "__main__":
    main()

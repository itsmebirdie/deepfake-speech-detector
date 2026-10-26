"""Generate leave-one-vocoder-out split files (PR-13). Owner: A7.

For each of the 6 vocoders, writes data/splits/lovo/<vocoder>/{train,dev,test}.txt:
  train/dev = bona fide + the other 5 vocoders (official train/dev utterances)
  test      = bona fide test clips + the held-out vocoder's fakes only (official test utterances)
No randomness: the folds are exact filters of the official split files.

Example (from the repo root):
  python scripts/make_lovo_splits.py
"""
import argparse
import os
from collections import Counter

from phasecon.data.splits import build_lovo_splits, read_split_file, write_split
from phasecon.utils.config import load_many


def main():
    ap = argparse.ArgumentParser(description="Generate leave-one-vocoder-out split files.")
    ap.add_argument("--config", nargs="+", default=["configs/default.yaml", "configs/data/librisevoc.yaml"],
                    help="one or more YAML configs (merged left to right)")
    ap.add_argument("--override", nargs="*", default=[], help="dotted overrides")
    args = ap.parse_args()

    cfg = load_many(args.config, args.override)
    data_cfg = cfg["data"]
    vocoders = data_cfg["vocoders"]
    out_dir = data_cfg["lovo"]["split_dir"]
    official = {
        split: read_split_file(data_cfg["split_files"][split], data_cfg["root"], vocoders)
        for split in ("train", "dev", "test")
    }
    for held_out in vocoders:
        folds = build_lovo_splits(official, held_out, vocoders)
        summary = []
        for split, entries in folds.items():
            path = os.path.join(out_dir, held_out, f"{split}.txt")
            header = (f"LibriSeVoc leave-one-vocoder-out fold, held_out={held_out}, split={split}\n"
                      f"built by scripts/make_lovo_splits.py from {data_cfg['split_files'][split]}")
            write_split(entries, path, header)
            n = Counter(e.label for e in entries)
            summary.append(f"{split} {len(entries)} (real {n[0]}, fake {n[1]})")
        print(f"held out {held_out:18s} " + " | ".join(summary))


if __name__ == "__main__":
    main()

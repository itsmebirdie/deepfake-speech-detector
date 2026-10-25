"""Build fixed class-balanced subset manifests (PR-09). Owner: A6.

Writes data/subsets/<split>.txt for every split whose `data.subset.per_vocoder_<split>` is set.
The files are small and committed, so the whole team trains on identical clips.

Example (from the repo root):
  python scripts/make_subset.py
"""
import argparse
import os
from collections import Counter

from phasecon.data.splits import read_split_file
from phasecon.data.subset import build_balanced_subset, save_manifest
from phasecon.utils.config import load_many


def main():
    ap = argparse.ArgumentParser(description="Build fixed class-balanced subset manifests.")
    ap.add_argument("--config", nargs="+", default=["configs/default.yaml", "configs/data/librisevoc.yaml"],
                    help="one or more YAML configs (merged left to right)")
    ap.add_argument("--override", nargs="*", default=[], help="dotted overrides, e.g. data.subset.per_vocoder_train=500")
    ap.add_argument("--check-files", action="store_true", help="also check every listed clip exists under data.root")
    args = ap.parse_args()

    cfg = load_many(args.config, args.override)
    data_cfg, seed = cfg["data"], cfg["seed"]
    subset = data_cfg["subset"]
    vocoders = data_cfg["vocoders"]
    for split in ("train", "dev", "test"):
        per_vocoder = subset.get(f"per_vocoder_{split}")
        if per_vocoder is None:
            print(f"{split}: per_vocoder_{split} is null -> full official split, no manifest")
            continue
        entries = read_split_file(data_cfg["split_files"][split], data_cfg["root"], vocoders,
                                  check_exists=args.check_files)
        picked = build_balanced_subset(entries, per_vocoder, seed)
        path = os.path.join(subset["manifest_dir"], f"{split}.txt")
        header = (f"LibriSeVoc balanced subset, split={split}, per_vocoder={per_vocoder}, seed={seed}\n"
                  f"built by scripts/make_subset.py from {data_cfg['split_files'][split]}")
        save_manifest(picked, path, header)
        counts = Counter(e.vocoder_id for e in picked)
        per_name = ", ".join(f"{v}={counts[i]}" for i, v in enumerate(vocoders, start=1))
        print(f"{split}: {len(picked)} clips, {len({e.utt_id for e in picked})} utterances, "
              f"real={counts[0]}, fake={len(picked) - counts[0]} ({per_name}) -> {path}")


if __name__ == "__main__":
    main()

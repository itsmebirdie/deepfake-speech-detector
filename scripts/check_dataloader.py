"""Iterate the real dataloader once over a split and report throughput (PR-04 evidence).

Example (from the repo root):
  python scripts/check_dataloader.py --split train
"""
import argparse
import time

from phasecon.data.librisevoc import build_dataloader
from phasecon.utils.config import load_many


def main():
    ap = argparse.ArgumentParser(description="Iterate the LibriSeVoc dataloader once and time it.")
    ap.add_argument(
        "--config",
        nargs="+",
        default=["configs/default.yaml", "configs/data/librisevoc.yaml", "configs/train/default.yaml"],
        help="YAML configs, merged left to right",
    )
    ap.add_argument("--override", nargs="*", default=[], help="dotted overrides, e.g. data.num_workers=0")
    ap.add_argument("--split", default="train", choices=["train", "dev", "test"])
    args = ap.parse_args()

    cfg = load_many(args.config, args.override)
    loader = build_dataloader(cfg, args.split, train=(args.split == "train"))
    total = len(loader.dataset)
    print(
        f"split={args.split} clips={total} batch_size={loader.batch_size} "
        f"num_workers={loader.num_workers} sample_rate={cfg['audio']['sample_rate']} "
        f"num_samples={cfg['audio']['num_samples']}",
        flush=True,
    )
    start = time.time()
    seen = 0
    for i, batch in enumerate(loader):
        seen += batch["waveform"].shape[0]
        if i == 0:
            print(
                f"first batch: waveform {tuple(batch['waveform'].shape)}, label {tuple(batch['label'].shape)}, "
                f"vocoder_id {tuple(batch['vocoder_id'].shape)}, utt_id list of {len(batch['utt_id'])}",
                flush=True,
            )
        if (i + 1) % 100 == 0:
            rate = seen / (time.time() - start)
            print(f"  {seen}/{total} clips, {rate:.0f} clips/s", flush=True)
    elapsed = time.time() - start
    print(f"DONE: {seen}/{total} clips in {elapsed / 60:.1f} min ({seen / elapsed:.0f} clips/s)", flush=True)


if __name__ == "__main__":
    main()

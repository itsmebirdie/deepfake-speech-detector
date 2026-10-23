"""Split-file parsing and leave-one-vocoder-out (LOVO) split construction. Owner: Member A."""
import os
from dataclasses import dataclass
from typing import Dict, List

REAL_DIR = "gt"  # folder of bona fide clips inside data.root


@dataclass
class Entry:
    path: str          # relative to data.root
    utt_id: str        # LibriTTS-style id (reader_chapter_...)
    label: int         # 0 bona fide, 1 synthetic
    vocoder_id: int    # 0 bona fide, 1..6 vocoders


def read_split_file(
    split_file: str, root: str, vocoders: List[str], check_exists: bool = False
) -> List[Entry]:
    """Parse an official train/dev/test .txt file into Entry objects (step A3).

    Each line of the split file names one bona fide clip, e.g. `27_123349_000006_000000.wav`.
    It expands to 1 + len(vocoders) entries:
      - the real clip   `gt/<name>.wav`                 label 0, vocoder_id 0
      - one fake per vocoder `<vocoder>/<name>_gen.wav`  label 1, vocoder_id 1..len(vocoders)
    `vocoder_id` follows the ORDER of `vocoders` (config `data.vocoders`; see DATASET_NOTES.md section 3).
    All 1 + len(vocoders) entries share the same `utt_id`, so a split never leaks an utterance.
    With `check_exists=True`, every file must exist under `root`.
    """
    if not vocoders:
        raise ValueError("vocoders is empty; set data.vocoders in configs/data/librisevoc.yaml")
    if len(set(vocoders)) != len(vocoders) or REAL_DIR in vocoders:
        raise ValueError(f"vocoders must be unique and must not include '{REAL_DIR}': {vocoders}")
    with open(split_file) as f:
        names = [line.strip() for line in f if line.strip()]
    if len(set(names)) != len(names):
        raise ValueError(f"{split_file} lists some clips more than once")
    entries: List[Entry] = []
    for name in names:
        if not name.endswith(".wav"):
            raise ValueError(f"unexpected line in {split_file}: {name!r} (expected a .wav name)")
        utt_id = name[: -len(".wav")]
        entries.append(Entry(path=f"{REAL_DIR}/{name}", utt_id=utt_id, label=0, vocoder_id=0))
        for vocoder_id, vocoder in enumerate(vocoders, start=1):
            entries.append(
                Entry(path=f"{vocoder}/{utt_id}_gen.wav", utt_id=utt_id, label=1, vocoder_id=vocoder_id)
            )
    if check_exists:
        missing = [e.path for e in entries if not os.path.exists(os.path.join(root, e.path))]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} clips from {split_file} are missing under {root}, e.g. {missing[:3]}"
            )
    return entries


def build_lovo_splits(
    entries_by_split: Dict[str, List[Entry]], held_out: str, vocoders: List[str]
) -> Dict[str, List[Entry]]:
    """LOVO protocol (step A7), to be written down in docs/DATASET_NOTES.md:
    - train/dev: bona fide + all vocoders EXCEPT `held_out`
    - test: bona fide test clips + `held_out` fakes only (report pooled and per-vocoder EER)
    Must assert zero overlap of utt_id between train/dev and test where the official split guarantees it.
    """
    raise NotImplementedError("TODO(A7)")


def write_split(entries: List[Entry], out_path: str) -> None:
    raise NotImplementedError("TODO(A7)")

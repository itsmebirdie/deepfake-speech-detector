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
    """LOVO protocol (step A7), written down in docs/DATASET_NOTES.md section 10:
    - train/dev: bona fide + all vocoders EXCEPT `held_out`
    - test: bona fide test clips + `held_out` fakes only (report pooled and per-vocoder EER)
    Must assert zero overlap of utt_id between train/dev and test where the official split guarantees it.

    `vocoder_id` keeps its global meaning (1..6 in `vocoders` order), so a held-out id simply never
    appears in train/dev and per-vocoder results stay comparable across folds.
    """
    if held_out not in vocoders:
        raise ValueError(f"held_out={held_out!r} is not one of {vocoders}")
    held_id = vocoders.index(held_out) + 1
    folds = {
        "train": [e for e in entries_by_split["train"] if e.vocoder_id != held_id],
        "dev": [e for e in entries_by_split["dev"] if e.vocoder_id != held_id],
        "test": [e for e in entries_by_split["test"] if e.vocoder_id in (0, held_id)],
    }
    for split in ("train", "dev"):
        if any(e.vocoder_id == held_id for e in folds[split]):
            raise AssertionError(f"held-out vocoder {held_out} leaked into {split}")
    if any(e.vocoder_id not in (0, held_id) for e in folds["test"]):
        raise AssertionError("LOVO test may only hold bona fide clips and held-out fakes")
    utts = {split: {e.utt_id for e in entries} for split, entries in folds.items()}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = utts[a] & utts[b]
        if overlap:
            raise AssertionError(f"{len(overlap)} utterances appear in both {a} and {b}, e.g. {sorted(overlap)[:3]}")
    return folds


def write_split(entries: List[Entry], out_path: str, header: str = "") -> None:
    """Write entries as a manifest (same format as data/subsets/*.txt; read back with load_manifest)."""
    from .subset import save_manifest  # local import: subset.py imports this module

    save_manifest(entries, out_path, header)

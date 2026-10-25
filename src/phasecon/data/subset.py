"""Fixed, class-balanced subsets per vocoder. Owner: Member A (step A6).

How the balanced subset is built (balance_real=True, the default):
  1. Shuffle the split's utterances with a fixed seed.
  2. Take the first 6 x per_vocoder utterances (6 = number of vocoders).
  3. Give each one its REAL clip plus exactly ONE fake, cycling through the vocoders
     (utterance 1 -> vocoder 1, utterance 2 -> vocoder 2, ...).
Result: per_vocoder fakes from every vocoder, the same number of real clips as fakes (50/50), and
every fake sits next to the real recording it was made from (same speaker, same words).

With balance_real=False it instead takes per_vocoder utterances with ALL their clips
(1 real + 6 fakes each): fewer utterances, 1 real : 6 fake, like the full dataset.
"""
import os
import random
from collections import defaultdict
from typing import Dict, List

from .splits import Entry


def _group_by_utterance(entries: List[Entry]) -> Dict[str, Dict[int, Entry]]:
    groups: Dict[str, Dict[int, Entry]] = defaultdict(dict)
    for e in entries:
        groups[e.utt_id][e.vocoder_id] = e
    return groups


def build_balanced_subset(
    entries: List[Entry], per_vocoder: int, seed: int, balance_real: bool = True
) -> List[Entry]:
    """Deterministically pick `per_vocoder` clips per vocoder and an equal TOTAL number of bona fide
    clips. Pairs (real clip + its resynthesis) should be kept together where possible so the
    'same speaker/content' property of LibriSeVoc is preserved."""
    if per_vocoder <= 0:
        raise ValueError(f"per_vocoder must be positive, got {per_vocoder}")
    groups = _group_by_utterance(entries)
    vocoder_ids = sorted({e.vocoder_id for e in entries if e.vocoder_id != 0})
    if not vocoder_ids:
        raise ValueError("entries contain no synthetic clips")
    utts = sorted(u for u, g in groups.items() if 0 in g and all(v in g for v in vocoder_ids))
    random.Random(seed).shuffle(utts)

    picked: List[Entry] = []
    if balance_real:
        needed = per_vocoder * len(vocoder_ids)
        if needed > len(utts):
            raise ValueError(
                f"per_vocoder={per_vocoder} needs {needed} utterances, but this split has {len(utts)} "
                f"(max per_vocoder = {len(utts) // len(vocoder_ids)})"
            )
        for i, utt in enumerate(utts[:needed]):
            vid = vocoder_ids[i % len(vocoder_ids)]
            picked += [groups[utt][0], groups[utt][vid]]
    else:
        if per_vocoder > len(utts):
            raise ValueError(f"per_vocoder={per_vocoder} but this split has only {len(utts)} utterances")
        for utt in utts[:per_vocoder]:
            picked += [groups[utt][v] for v in [0] + vocoder_ids]
    return picked


def save_manifest(entries: List[Entry], path: str, header: str = "") -> None:
    """One clip per line, tab-separated: path, utt_id, label, vocoder_id. Lines starting with # are comments."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        for line in header.splitlines():
            f.write(f"# {line}\n")
        f.write("# path\tutt_id\tlabel\tvocoder_id\n")
        for e in entries:
            f.write(f"{e.path}\t{e.utt_id}\t{e.label}\t{e.vocoder_id}\n")


def load_manifest(path: str) -> List[Entry]:
    """Read a manifest written by `save_manifest`."""
    entries: List[Entry] = []
    with open(path) as f:
        for n, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                raise ValueError(f"{path} line {n}: expected 4 tab-separated fields, got {len(parts)}")
            p, utt, label, vid = parts
            entries.append(Entry(path=p, utt_id=utt, label=int(label), vocoder_id=int(vid)))
    return entries

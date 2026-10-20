"""Owner: Member A. Tests for the data pipeline (PR-04). Subset and LOVO tests stay skipped until PR-09 / PR-13.

These tests never touch the real 42 GB dataset: the parsing and overlap tests read the committed split
files, and the loading tests build a tiny fake LibriSeVoc folder in a temporary directory.
"""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sf = pytest.importorskip("soundfile")
yaml = pytest.importorskip("yaml")

from phasecon.data.collate import collate_fn  # noqa: E402
from phasecon.data.librisevoc import LibriSeVocDataset, build_dataloader  # noqa: E402
from phasecon.data.splits import read_split_file  # noqa: E402
from phasecon.data.transforms import pad_or_crop  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SPLITS = REPO / "data" / "splits"
VOCODERS = yaml.safe_load(open(REPO / "configs" / "data" / "librisevoc.yaml"))["data"]["vocoders"]
SPLIT_SIZES = {"train": 7920, "dev": 2640, "test": 2641}


def baseline_pad(x: np.ndarray, max_len: int) -> np.ndarray:
    """Copy of `pad()` from the authors' main.py, used as the reference."""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, (1, num_repeats))[:, :max_len][0]


def make_fake_librisevoc(root: Path, utt_ids, seconds, rate=24000):
    """Write a tiny LibriSeVoc-shaped folder: gt/ plus one folder per vocoder."""
    rng = np.random.default_rng(0)
    for utt, sec in zip(utt_ids, seconds):
        n = int(sec * rate)
        for folder, name in [("gt", f"{utt}.wav")] + [(v, f"{utt}_gen.wav") for v in VOCODERS]:
            (root / folder).mkdir(parents=True, exist_ok=True)
            sf.write(root / folder / name, (rng.standard_normal(n) * 0.1).astype(np.float32), rate)


def read_split_file_from_names(root: Path, names):
    split = root / "split.txt"
    split.write_text("\n".join(names) + "\n")
    return read_split_file(str(split), str(root), VOCODERS, check_exists=True)


def test_entry_parsing():
    """Official split files parse; label and vocoder_id mapping match folder names."""
    assert len(VOCODERS) == 6
    for split, size in SPLIT_SIZES.items():
        entries = read_split_file(str(SPLITS / f"{split}.txt"), "unused", VOCODERS)
        assert len(entries) == size * 7
        first = entries[:7]
        assert first[0].path.startswith("gt/") and first[0].label == 0 and first[0].vocoder_id == 0
        for vocoder_id, (entry, vocoder) in enumerate(zip(first[1:], VOCODERS), start=1):
            assert entry.path == f"{vocoder}/{first[0].utt_id}_gen.wav"
            assert entry.label == 1 and entry.vocoder_id == vocoder_id
            assert entry.utt_id == first[0].utt_id
        assert {e.vocoder_id for e in entries} == set(range(7))
        assert {e.label for e in entries} == {0, 1}


def test_entry_parsing_rejects_bad_vocoders():
    with pytest.raises(ValueError):
        read_split_file(str(SPLITS / "dev.txt"), "unused", [])
    with pytest.raises(ValueError):
        read_split_file(str(SPLITS / "dev.txt"), "unused", ["gt", "melgan"])


def test_item_shapes(tmp_path):
    """__getitem__ returns waveform [num_samples] float32, scalar label/vocoder_id."""
    make_fake_librisevoc(tmp_path, ["1_2_000001_000001", "1_2_000002_000001"], [6.0, 2.5])
    split = tmp_path / "split.txt"
    split.write_text("1_2_000001_000001.wav\n1_2_000002_000001.wav\n")
    entries = read_split_file(str(split), str(tmp_path), VOCODERS, check_exists=True)
    for rate, num_samples in [(24000, 96000), (16000, 64000)]:  # native rate, and resampled
        ds = LibriSeVocDataset(entries, num_samples=num_samples, sample_rate=rate, root=str(tmp_path))
        assert len(ds) == 14
        for idx in (0, 3, 7):  # long real clip, long fake clip, short real clip
            item = ds[idx]
            assert item["waveform"].shape == (num_samples,)
            assert item["waveform"].dtype == torch.float32
            assert item["label"].dtype == torch.long and item["label"].dim() == 0
            assert item["vocoder_id"].dtype == torch.long and item["vocoder_id"].dim() == 0
            assert item["label"].item() == entries[idx].label
            assert item["vocoder_id"].item() == entries[idx].vocoder_id
            assert isinstance(item["utt_id"], str)


def test_item_matches_file_start(tmp_path):
    """At the native rate, a long clip's waveform is exactly the first num_samples of the file."""
    make_fake_librisevoc(tmp_path, ["1_2_000001_000001"], [6.0])
    entries = read_split_file_from_names(tmp_path, ["1_2_000001_000001.wav"])
    ds = LibriSeVocDataset(entries, num_samples=96000, sample_rate=24000, root=str(tmp_path))
    audio, _ = sf.read(tmp_path / "gt" / "1_2_000001_000001.wav", dtype="float32")
    assert torch.equal(ds[0]["waveform"], torch.from_numpy(audio[:96000]))


def test_missing_files_are_reported(tmp_path):
    split = tmp_path / "split.txt"
    split.write_text("9_9_000001_000001.wav\n")
    with pytest.raises(FileNotFoundError):
        read_split_file(str(split), str(tmp_path), VOCODERS, check_exists=True)


def test_pad_or_crop():
    """Short clips repeat-padded, long clips cropped to exactly num_samples."""
    rng = np.random.default_rng(1)
    for length in (1, 7, 50_000, 96_000, 150_000):
        x = rng.standard_normal(length).astype(np.float32)
        out = pad_or_crop(torch.from_numpy(x), 96_000, random_crop=False)
        assert out.shape == (96_000,)
        assert np.array_equal(out.numpy(), baseline_pad(x, 96_000))
    long = torch.arange(150_000, dtype=torch.float32)
    for _ in range(5):
        out = pad_or_crop(long, 96_000, random_crop=True)
        assert out.shape == (96_000,)
        assert torch.equal(out, torch.arange(out[0].item(), out[0].item() + 96_000))
    with pytest.raises(ValueError):
        pad_or_crop(torch.zeros(0), 10, random_crop=False)
    with pytest.raises(ValueError):
        pad_or_crop(torch.zeros(2, 10), 10, random_crop=False)


def test_no_utt_overlap_between_splits():
    """train/dev/test share no utt_id (official split)."""
    ids = {
        split: {e.utt_id for e in read_split_file(str(SPLITS / f"{split}.txt"), "unused", VOCODERS)}
        for split in SPLIT_SIZES
    }
    assert {k: len(v) for k, v in ids.items()} == SPLIT_SIZES
    assert not ids["train"] & ids["dev"]
    assert not ids["train"] & ids["test"]
    assert not ids["dev"] & ids["test"]


def test_balanced_subset_is_deterministic():
    """Same seed -> identical manifest; per-vocoder counts exact; classes balanced."""
    pytest.skip("not implemented yet (PR-09)")


def test_lovo_split_excludes_held_out():
    """Held-out vocoder absent from train/dev, only held-out fakes + real in test."""
    pytest.skip("not implemented yet (PR-13)")


def test_collate_shapes():
    """collate_fn yields [B,T], [B], [B], list[str]."""
    batch = [
        {
            "waveform": torch.randn(100),
            "label": torch.tensor(i % 2),
            "vocoder_id": torch.tensor(i),
            "utt_id": f"u{i}",
        }
        for i in range(4)
    ]
    out = collate_fn(batch)
    assert out["waveform"].shape == (4, 100)
    assert out["label"].shape == (4,) and out["vocoder_id"].shape == (4,)
    assert out["utt_id"] == ["u0", "u1", "u2", "u3"]
    with pytest.raises(ValueError):
        collate_fn([])


def test_build_dataloader_iterates(tmp_path):
    """build_dataloader turns a config into batches with the INTERFACES.md section 1 shapes."""
    names = [f"1_2_00000{i}_000001" for i in range(3)]
    make_fake_librisevoc(tmp_path, names, [5.0, 4.5, 3.0])
    split = tmp_path / "split.txt"
    split.write_text("\n".join(n + ".wav" for n in names) + "\n")
    cfg = {
        "seed": 1234,
        "audio": {"sample_rate": 24000, "num_samples": 96000},
        "data": {
            "root": str(tmp_path),
            "split_files": {"train": str(split)},
            "vocoders": VOCODERS,
            "num_workers": 0,
        },
        "train": {"batch_size": 8},
    }
    loader = build_dataloader(cfg, "train", train=True)
    batches = list(loader)
    assert [b["waveform"].shape[0] for b in batches] == [8, 8, 5]
    assert all(b["waveform"].shape[1] == 96000 for b in batches)
    seen = sorted((u, v.item()) for b in batches for u, v in zip(b["utt_id"], b["vocoder_id"]))
    assert seen == sorted((n, v) for n in names for v in range(7))
    order = [b["utt_id"] for b in build_dataloader(cfg, "train", train=True)]
    assert order == [b["utt_id"] for b in batches]  # seeded shuffle is repeatable

"""Owner: Member A. Tests are specified but skipped until the module is implemented."""
import pytest


def test_entry_parsing():
    """Official split files parse; label and vocoder_id mapping match folder names."""
    pytest.skip("not implemented yet")


def test_item_shapes():
    """__getitem__ returns waveform [num_samples] float32, scalar label/vocoder_id."""
    pytest.skip("not implemented yet")


def test_pad_or_crop():
    """Short clips repeat-padded, long clips cropped to exactly num_samples."""
    pytest.skip("not implemented yet")


def test_no_utt_overlap_between_splits():
    """train/dev/test share no utt_id (official split)."""
    pytest.skip("not implemented yet")


def test_balanced_subset_is_deterministic():
    """Same seed -> identical manifest; per-vocoder counts exact; classes balanced."""
    pytest.skip("not implemented yet")


def test_lovo_split_excludes_held_out():
    """Held-out vocoder absent from train/dev, only held-out fakes + real in test."""
    pytest.skip("not implemented yet")


def test_collate_shapes():
    """collate_fn yields [B,T], [B], [B], list[str]."""
    pytest.skip("not implemented yet")

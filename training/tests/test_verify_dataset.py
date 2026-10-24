"""Dataset checksum must separate file names from contents."""

from __future__ import annotations

from pathlib import Path

from training.scripts.verify_dataset import hash_directory


def test_hash_distinguishes_name_content_boundary(tmp_path: Path) -> None:
    """File ``a`` holding ``bc`` must not hash like file ``ab`` holding ``c``."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a").write_bytes(b"bc")
    (second / "ab").write_bytes(b"c")
    assert hash_directory(first) != hash_directory(second)

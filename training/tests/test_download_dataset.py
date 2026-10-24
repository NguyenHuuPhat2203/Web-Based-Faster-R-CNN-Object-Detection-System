"""Bounded zip extraction must reject path traversal."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from training.scripts.download_dataset import safe_extract_zip


def test_safe_extract_zip_rejects_parent_escape(tmp_path: Path) -> None:
    """A ``../evil.txt`` member must raise and must not be written outside dest."""
    dest = tmp_path / "out"
    dest.mkdir()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../evil.txt", "pwned")
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        with pytest.raises(ValueError):
            safe_extract_zip(zf, dest)
    assert not (tmp_path / "evil.txt").exists()

#!/usr/bin/env python3
"""Verify dataset directory hash against the expected checksum file."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def hash_directory(root: Path) -> str:
    """Compute a deterministic SHA-256 over relative file paths and contents.

    Files are ordered by POSIX relative path (platform-independent) and each
    path and content is length-prefixed so boundaries cannot be confused.
    """
    files = sorted(
        (p.relative_to(root).as_posix(), p) for p in root.rglob("*") if p.is_file()
    )
    digest = hashlib.sha256()
    for rel, path in files:
        encoded = rel.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as fh:
            while chunk := fh.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def write_checksum(
    dataset_dir: Path, checksum_path: Path | None = None
) -> tuple[Path, str]:
    """Hash ``dataset_dir`` and write the ``.sha256`` file; return (path, digest)."""
    checksum_path = checksum_path or Path(f"{dataset_dir}.sha256")
    digest = hash_directory(dataset_dir)
    checksum_path.write_text(digest + "\n", encoding="utf-8")
    return checksum_path, digest


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Dataset root, e.g. data/raw/v1",
    )
    parser.add_argument(
        "--checksum",
        type=Path,
        help="Path to .sha256 file (default: <dataset-dir>.sha256)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write checksum file instead of verifying",
    )
    args = parser.parse_args(argv)

    dataset_dir: Path = args.dataset_dir
    if not dataset_dir.is_dir():
        print(f"Dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 1

    if args.write:
        checksum_path, actual = write_checksum(dataset_dir, args.checksum)
        print(f"Wrote {checksum_path}: {actual}")
        return 0

    checksum_path = args.checksum or Path(f"{dataset_dir}.sha256")
    actual = hash_directory(dataset_dir)

    if not checksum_path.exists():
        print(f"Checksum file missing: {checksum_path}", file=sys.stderr)
        print(f"Actual hash: {actual}")
        return 1

    expected = checksum_path.read_text(encoding="utf-8").strip()
    if actual != expected:
        print("Dataset hash mismatch!", file=sys.stderr)
        print(f"expected={expected}", file=sys.stderr)
        print(f"actual={actual}", file=sys.stderr)
        return 1

    print(f"Dataset OK: {dataset_dir} hash={actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

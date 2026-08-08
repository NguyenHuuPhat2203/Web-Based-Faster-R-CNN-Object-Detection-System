#!/usr/bin/env python3
"""Verify dataset directory hash against the expected checksum file."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def hash_directory(root: Path) -> str:
    """Compute a deterministic SHA-256 over relative file paths and contents."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(path.read_bytes())
    return digest.hexdigest()


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

    checksum_path = args.checksum or Path(f"{dataset_dir}.sha256")
    actual = hash_directory(dataset_dir)

    if args.write:
        checksum_path.write_text(actual + "\n", encoding="utf-8")
        print(f"Wrote {checksum_path}: {actual}")
        return 0

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

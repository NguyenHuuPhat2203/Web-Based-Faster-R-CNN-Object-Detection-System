#!/usr/bin/env python3
"""Download the Ultralytics brain-tumor dataset and normalize into data/raw/v1."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from training.data.layout import list_images, write_dataset_yaml
from training.scripts.verify_dataset import write_checksum

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ZIP_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/brain-tumor.zip"
)
_DOWNLOAD_CHUNK = 64 * 1024
_MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024
_MAX_ZIP_ENTRIES = 4096
_MAX_ZIP_UNCOMPRESSED = 2 * 1024 * 1024 * 1024
_MAX_ZIP_RATIO = 100

KAGGLE_FALLBACK = """
Download failed from Ultralytics assets.

Fallback (Kaggle / manual):
  1. See https://docs.ultralytics.com/datasets/detect/brain-tumor/
  2. Or: kaggle datasets download -d ultralytics/brain-tumor
  3. Extract so you have YOLO images + labels, then re-run with --force
     after placing files under a temp dir and pointing --zip-path at the zip.

Primary URL:
  {url}
""".strip()


def safe_extract_zip(zf: zipfile.ZipFile, dest: Path | str) -> None:
    """Extract ``zf`` into ``dest`` with zip-slip / zip-bomb guards."""
    dest_root = Path(dest).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    infos = zf.infolist()
    if len(infos) > _MAX_ZIP_ENTRIES:
        raise ValueError("zip has too many entries")
    total_size = 0
    for info in infos:
        name = info.filename.replace("\\", "/")
        if name.startswith("/") or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"unsafe zip member: {info.filename}")
        if (
            info.compress_size > 0
            and info.file_size / info.compress_size > _MAX_ZIP_RATIO
        ):
            raise ValueError(f"zip bomb ratio: {info.filename}")
        total_size += info.file_size
        if total_size > _MAX_ZIP_UNCOMPRESSED:
            raise ValueError("zip uncompressed size exceeds 2GiB")
        target = (dest_root / name).resolve()
        if not target.is_relative_to(dest_root):
            raise ValueError(f"unsafe zip member: {info.filename}")
        zf.extract(info, dest_root)


def _stream_download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` in 64KiB chunks, aborting over 512MiB."""
    with urllib.request.urlopen(url, timeout=120) as response, dest.open("wb") as fh:
        total = 0
        while True:
            chunk = response.read(_DOWNLOAD_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_DOWNLOAD_BYTES:
                raise ValueError("download exceeds 512MiB")
            fh.write(chunk)


def is_valid_layout(root: Path) -> bool:
    """Return True when normalized train/valid image dirs exist and are non-empty."""
    train_images = root / "train" / "images"
    valid_images = root / "valid" / "images"
    if not train_images.is_dir() or not valid_images.is_dir():
        return False
    return bool(list_images(train_images)) and bool(list_images(valid_images))


def _find_split_dirs(extracted: Path) -> tuple[Path, Path, Path | None, Path | None]:
    """Locate train/val image dirs and optional label dirs inside extracted zip."""
    candidates = [
        (
            extracted / "images" / "train",
            extracted / "images" / "val",
            extracted / "labels" / "train",
            extracted / "labels" / "val",
        ),
        (
            extracted / "train" / "images",
            extracted / "valid" / "images",
            extracted / "train" / "labels",
            extracted / "valid" / "labels",
        ),
        (
            extracted / "train" / "images",
            extracted / "val" / "images",
            extracted / "train" / "labels",
            extracted / "val" / "labels",
        ),
    ]
    # Also search one level deeper (zip root folder)
    for child in extracted.iterdir():
        if child.is_dir():
            candidates.extend(
                [
                    (
                        child / "images" / "train",
                        child / "images" / "val",
                        child / "labels" / "train",
                        child / "labels" / "val",
                    ),
                    (
                        child / "train" / "images",
                        child / "valid" / "images",
                        child / "train" / "labels",
                        child / "valid" / "labels",
                    ),
                ]
            )

    for train_img, val_img, train_lbl, val_lbl in candidates:
        if train_img.is_dir() and val_img.is_dir():
            return (
                train_img,
                val_img,
                train_lbl if train_lbl.is_dir() else None,
                val_lbl if val_lbl.is_dir() else None,
            )
    raise FileNotFoundError(
        f"Could not find images/train and images/val under {extracted}"
    )


def _copy_split(
    *,
    src_images: Path,
    src_labels: Path | None,
    dest_images: Path,
    dest_labels: Path,
) -> int:
    """Copy images and matching label txt files into dest dirs."""
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)
    count = 0
    for image_path in list_images(src_images):
        shutil.copy2(image_path, dest_images / image_path.name)
        label_name = image_path.stem + ".txt"
        label_src: Path | None = None
        if src_labels is not None:
            candidate = src_labels / label_name
            if candidate.exists():
                label_src = candidate
        sidecar = src_images / label_name
        if label_src is None and sidecar.exists():
            label_src = sidecar
        if label_src is not None:
            shutil.copy2(label_src, dest_labels / label_name)
        else:
            # Empty label file = background-only image
            (dest_labels / label_name).write_text("", encoding="utf-8")
        count += 1
    return count


def download_and_normalize(
    *,
    dest: Path,
    zip_url: str = DATASET_ZIP_URL,
    zip_path: Path | None = None,
    force: bool = False,
) -> Path:
    """Download (or use local zip), normalize layout, write yaml, return dest."""
    if is_valid_layout(dest) and not force:
        print(f"Dataset already present at {dest} (use --force to re-download)")
        return dest

    if dest.exists() and force:
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    work = dest.parent / ".download_tmp"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    try:
        if zip_path is not None:
            if not zip_path.exists():
                raise FileNotFoundError(f"Local zip not found: {zip_path}")
            local_zip = zip_path
        else:
            local_zip = work / "brain-tumor.zip"
            print(f"Downloading {zip_url} ...")
            try:
                _stream_download(zip_url, local_zip)
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                print(KAGGLE_FALLBACK.format(url=zip_url), file=sys.stderr)
                raise SystemExit(1) from exc

        extract_dir = work / "extracted"
        extract_dir.mkdir()
        with zipfile.ZipFile(local_zip, "r") as zf:
            safe_extract_zip(zf, extract_dir)

        train_img, val_img, train_lbl, val_lbl = _find_split_dirs(extract_dir)
        n_train = _copy_split(
            src_images=train_img,
            src_labels=train_lbl,
            dest_images=dest / "train" / "images",
            dest_labels=dest / "train" / "labels",
        )
        n_val = _copy_split(
            src_images=val_img,
            src_labels=val_lbl,
            dest_images=dest / "valid" / "images",
            dest_labels=dest / "valid" / "labels",
        )
        write_dataset_yaml(dest / "brain-tumor.yaml")
        print(f"Normalized dataset -> {dest} (train={n_train}, valid={n_val})")
    finally:
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)

    return dest


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=_PROJECT_ROOT / "data" / "raw" / "v1",
    )
    parser.add_argument("--zip-url", default=DATASET_ZIP_URL)
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Use a local zip instead of downloading",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Do not write data/raw/v1.sha256",
    )
    args = parser.parse_args(argv)

    download_and_normalize(
        dest=args.dest,
        zip_url=args.zip_url,
        zip_path=args.zip_path,
        force=args.force,
    )

    if not args.skip_checksum:
        checksum_path, digest = write_checksum(args.dest)
        print(f"Wrote {checksum_path}: {digest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

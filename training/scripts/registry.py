"""Registry writer: the one place a model entry lands in the model registry.

All promote paths (Faster R-CNN checkpoint / Hugging Face / YOLO) write
through this module so the version guard, git-commit stamp, and metadata
shape live once. Sources vary only in how weights are converted.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRODUCTION_VERSION = "v1.0.0"
_VERSION_RE = re.compile(r"^v[0-9]+(?:\.[0-9]+){1,2}(?:-[A-Za-z0-9.]+)?$")


class RegistryGuardError(ValueError):
    """Raised when a promote would overwrite the production slot unconfirmed."""


def _git_commit() -> str:
    """Return current git commit hash or 'unknown'."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def ensure_promotable(version: str, *, i_know: bool) -> None:
    """Validate a registry version name and guard the production slot.

    Raises:
        RegistryGuardError: If the version is malformed, or is the production
            version and ``i_know`` is false.
    """
    if not _VERSION_RE.fullmatch(version):
        raise RegistryGuardError(f"Invalid registry version: {version}")
    if version == PRODUCTION_VERSION and not i_know:
        raise RegistryGuardError(
            f"Refusing to promote to {PRODUCTION_VERSION} without --i-know "
            "(use a smoke version, or pass --i-know)."
        )


def write_registry_entry(
    *,
    model_key: str,
    version: str,
    dataset_version: str,
    dataset_hash: str,
    metrics: dict[str, float | None],
    weights_src: Path,
    weights_name: str,
    convert: Callable[[Path, Path], None] | None = None,
    registry_root: Path,
    i_know: bool = False,
) -> Path:
    """Write one model entry into the registry; returns the weights path.

    Args:
        model_key: Registry subdirectory, e.g. ``"faster_rcnn"`` or ``"yolo"``.
        version: Version directory name (``v1.0.0`` requires ``i_know``).
        dataset_version: Dataset version label recorded in metadata.
        dataset_hash: Dataset hash recorded in metadata.
        metrics: Metric name/value pairs recorded under ``metadata.metrics``.
        weights_src: Source weights file.
        weights_name: Destination filename in the registry (e.g. ``model.safetensors``).
        convert: Optional converter ``(src, dest)``; ``None`` copies as-is.
        registry_root: Registry root directory.
        i_know: Explicit confirmation to write the production slot.

    Raises:
        RegistryGuardError: If the version is unsafe or would overwrite
            production without ``i_know``.
    """
    ensure_promotable(version, i_know=i_know)

    root = registry_root.resolve()
    out_dir = (root / model_key / version).resolve()
    if not out_dir.is_relative_to(root):
        raise RegistryGuardError(f"Registry path escapes root: {out_dir} (root={root})")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / weights_name
    if convert is None:
        shutil.copy2(weights_src, out_path)
    else:
        convert(weights_src, out_path)

    with out_path.open("rb") as fh:
        weights_sha256 = hashlib.file_digest(fh, "sha256").hexdigest()

    metadata = {
        "model_type": model_key,
        "version": version,
        "dataset_version": dataset_version,
        "dataset_hash": dataset_hash,
        "metrics": metrics,
        "git_commit": _git_commit(),
        "created_at": datetime.now(UTC).isoformat(),
        "weights": out_path.name,
        "weights_sha256": weights_sha256,
        "source_checkpoint": str(weights_src),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return out_path

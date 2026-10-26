#!/usr/bin/env python3
"""Sync every model weight file in the system with one Hugging Face model repo.

The Hub repo mirrors the local paths of the gitignored weights: ``registry/``
(served Faster R-CNN and YOLO), ``checkpoints/`` (loader fallback), and
``.weights/`` (SwinUNETR, MedNeXt). ``models.lock.json`` is committed and pins
the repo id plus the Hub commit that matches this code.

Settings come from ``.env`` (template: ``.env.example``); real environment
variables win. ``HF_TOKEN`` authenticates (write access for push) and
``HF_MODEL_REPO_ID`` names the repo for push. ``hf auth login`` and
``--repo-id`` work as well.

    python -m scripts.sync_models push --dry-run   # target repo and files
    python -m scripts.sync_models push
    python -m scripts.sync_models pull
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download
from huggingface_hub.utils import filter_repo_objects

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = PROJECT_ROOT / "models.lock.json"
ENV_PATH = PROJECT_ROOT / ".env"
ARTIFACT_DIRS = ("registry", "checkpoints", ".weights")
WEIGHT_SUFFIXES = (".safetensors", ".pt", ".pth")
# One filter for both directions: push uploads only these local files, and pull
# never writes the Hub's README.md or .gitattributes over the project's own.
ARTIFACT_PATTERNS = [f"{d}/*{s}" for d in ARTIFACT_DIRS for s in WEIGHT_SUFFIXES]


def select_artifacts(paths: Iterable[str]) -> list[str]:
    """Return the repo-relative POSIX paths that are synced weight files."""
    return sorted(filter_repo_objects(paths, allow_patterns=ARTIFACT_PATTERNS))


def local_artifacts(root: Path) -> list[str]:
    """Return the weight files under ``root`` that a push uploads."""
    return select_artifacts(
        path.relative_to(root).as_posix()
        for directory in ARTIFACT_DIRS
        for path in (root / directory).rglob("*")
        if path.is_file()
    )


def _read_lock() -> dict[str, str]:
    if not LOCK_PATH.is_file():
        return {}
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def _write_lock(repo_id: str, revision: str) -> None:
    lock = {"repo_id": repo_id, "revision": revision}
    LOCK_PATH.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")


def push(repo_id: str | None, message: str, dry_run: bool) -> int:
    """Upload local weights in one commit and pin that commit in the lock file.

    Files identical to the Hub copy are skipped by ``create_commit``; nothing is
    deleted remotely. A private repo is created when missing.
    """
    print(f"Repo: {repo_id or '(not set)'}")
    files = local_artifacts(PROJECT_ROOT)
    for name in files:
        print(f"{(PROJECT_ROOT / name).stat().st_size / 1e6:9.1f} MB  {name}")
    if not files:
        print("No weight files found", file=sys.stderr)
        return 1
    if dry_run:
        return 0
    if not repo_id:
        print("Set HF_MODEL_REPO_ID in .env or pass --repo-id", file=sys.stderr)
        return 2

    api = HfApi()
    api.create_repo(repo_id, private=True, exist_ok=True)
    commit = api.create_commit(
        repo_id,
        operations=[
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=PROJECT_ROOT / name)
            for name in files
        ],
        commit_message=message,
    )
    _write_lock(repo_id, commit.oid)
    print(f"Pinned {repo_id}@{commit.oid} in {LOCK_PATH.name}; commit that file.")
    return 0


def pull(repo_id: str, revision: str) -> int:
    """Make local weights match ``revision``; differing local copies are replaced."""
    snapshot_download(
        repo_id,
        revision=revision,
        local_dir=PROJECT_ROOT,
        allow_patterns=ARTIFACT_PATTERNS,
    )
    print(f"Pulled {repo_id}@{revision}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    load_dotenv(ENV_PATH)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    push_parser = commands.add_parser(
        "push", help="upload changed weights and pin the new commit"
    )
    push_parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_MODEL_REPO_ID") or None,
        help=f"<user>/<repo>; defaults to HF_MODEL_REPO_ID, then {LOCK_PATH.name}",
    )
    push_parser.add_argument("-m", "--message", default="Sync model weights")
    push_parser.add_argument(
        "--dry-run", action="store_true", help="list files, upload nothing"
    )
    pull_parser = commands.add_parser("pull", help="download the pinned weights")
    pull_parser.add_argument(
        "--revision", help=f"commit, tag, or branch; defaults to {LOCK_PATH.name}"
    )
    args = parser.parse_args(argv)

    lock = _read_lock()
    if args.command == "push":
        return push(args.repo_id or lock.get("repo_id"), args.message, args.dry_run)
    if "repo_id" not in lock:
        print(f"{LOCK_PATH.name} not found; push first", file=sys.stderr)
        return 2
    return pull(lock["repo_id"], args.revision or lock["revision"])


if __name__ == "__main__":
    raise SystemExit(main())

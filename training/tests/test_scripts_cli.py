"""CLI smoke tests for training scripts (no download / no training)."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "training.scripts.download_dataset",
        "training.scripts.subsample_dataset",
        "training.scripts.promote_checkpoint",
        "training.scripts.run_e2e_smoke",
        "training.scripts.verify_dataset",
    ],
)
def test_training_script_help(module_name: str) -> None:
    """Each training CLI must expose ``main`` and accept ``--help``."""
    import importlib

    module = importlib.import_module(module_name)
    with pytest.raises(SystemExit) as exc_info:
        module.main(["--help"])
    assert exc_info.value.code == 0


def test_promote_checkpoint_refuses_v1_without_flag() -> None:
    """Production registry slot requires an explicit confirmation flag."""
    from pathlib import Path

    from training.scripts.promote_checkpoint import main

    code = main(
        [
            "--model",
            "faster_rcnn",
            "--checkpoint",
            str(Path("nonexistent.pth")),
            "--version",
            "v1.0.0",
        ]
    )
    assert code == 2

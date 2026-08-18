"""BraTS tumor-region registry: label ids, names, display colors, channel decode.

Single source for the 1=NCR / 2=ED / 4=ET vocabulary shared by model decode
(``ml/backbone/segment3d``) and case analysis (``backend.services.case_store``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    """One BraTS tumor region."""

    label: int
    name: str
    color: tuple[int, int, int]
    channel: int


REGIONS: tuple[Region, ...] = (
    Region(label=1, name="NCR", color=(245, 158, 11), channel=0),
    Region(label=2, name="ED", color=(16, 185, 129), channel=1),
    Region(label=4, name="ET", color=(239, 68, 68), channel=2),
)

BY_LABEL: dict[int, Region] = {r.label: r for r in REGIONS}
BY_CHANNEL: dict[int, Region] = {r.channel: r for r in REGIONS}

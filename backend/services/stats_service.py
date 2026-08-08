"""User statistics aggregation."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from ..models import Detection, Image, User
from ..schemas.auth import UserStats


class StatsService:
    """Compute dashboard statistics for a user."""

    def __init__(self, db: Session) -> None:
        """Initialize the service with a database session."""
        self.db = db

    def get_user_stats(self, current_user: User) -> UserStats:
        """Return scan and detection counts for a user.

        Args:
            current_user: Authenticated user instance.

        Returns:
            Aggregated user statistics.
        """
        now = datetime.now(UTC)

        total_scans = (
            self.db.query(Image)
            .filter(Image.user_id == current_user.id, ~Image.deleted)
            .count()
        )

        scans_this_month = (
            self.db.query(Image)
            .filter(
                Image.user_id == current_user.id,
                ~Image.deleted,
                Image.uploaded_at >= now.replace(day=1),
            )
            .count()
        )

        total_detections = (
            self.db.query(Detection)
            .join(Image)
            .filter(Image.user_id == current_user.id, ~Image.deleted)
            .count()
        )

        return UserStats(
            total_scans=total_scans,
            total_detections=total_detections,
            scans_this_month=scans_this_month,
        )

"""Add composite indexes for common queries

Revision ID: 003
Revises: 002
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_images_user_id_deleted_uploaded_at",
        "images",
        ["user_id", "deleted", sa.text("uploaded_at DESC")],
    )
    op.create_index("ix_images_user_id_deleted", "images", ["user_id", "deleted"])
    op.create_index(
        "ix_detections_image_id_score",
        "detections",
        ["image_id", sa.text("score DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_detections_image_id_score")
    op.drop_index("ix_images_user_id_deleted")
    op.drop_index("ix_images_user_id_deleted_uploaded_at")

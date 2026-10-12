"""Add image metadata, detections table, migrate JSON data

Revision ID: 002
Revises: 001
"""
import json
from typing import Union, Sequence
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add new columns to images (nullable so existing rows pass)
    op.add_column("images", sa.Column("width", sa.Integer(), nullable=True))
    op.add_column("images", sa.Column("height", sa.Integer(), nullable=True))
    op.add_column("images", sa.Column("file_size", sa.Integer(), nullable=True))
    op.add_column("images", sa.Column("content_hash", sa.String(64), nullable=True))
    op.add_column("images", sa.Column("model_type", sa.String(50), nullable=True))
    op.add_column("images", sa.Column("model_version", sa.String(100), nullable=True))
    op.add_column("images", sa.Column("threshold", sa.Float(), nullable=True))

    # Create detections table
    op.create_table("detections",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("image_id", sa.Integer(), nullable=False),
        sa.Column("label", sa.Integer(), nullable=False),
        sa.Column("label_name", sa.String(100), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("x1", sa.Float(), nullable=False),
        sa.Column("y1", sa.Float(), nullable=False),
        sa.Column("x2", sa.Float(), nullable=False),
        sa.Column("y2", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["image_id"], ["images.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_detections_id", "detections", ["id"], unique=False)
    op.create_index("ix_detections_image_id", "detections", ["image_id"], unique=False)

    # Migrate existing JSON data from detection_result to detections rows
    conn = op.get_bind()
    rows = conn.execute(
        text("SELECT id, detection_result FROM images WHERE detection_result IS NOT NULL")
    ).fetchall()

    for row in rows:
        img_id, result_json = row
        try:
            result = json.loads(result_json) if isinstance(result_json, str) else result_json
        except (json.JSONDecodeError, TypeError):
            continue

        boxes = result.get("boxes", [])
        labels = result.get("labels", [])
        scores = result.get("scores", [])
        label_names = result.get("label_names", [])

        for i, box in enumerate(boxes):
            label_id = labels[i] if i < len(labels) else 0
            label_name = label_names[i] if i < len(label_names) else f"Class {label_id}"
            score = scores[i] if i < len(scores) else 0.0

            conn.execute(
                text("""INSERT INTO detections (image_id, label, label_name, score, x1, y1, x2, y2)
                       VALUES (:img_id, :label, :label_name, :score, :x1, :y1, :x2, :y2)"""),
                {
                    "img_id": img_id,
                    "label": label_id,
                    "label_name": label_name,
                    "score": score,
                    "x1": box[0], "y1": box[1],
                    "x2": box[2], "y2": box[3],
                },
            )

    # Drop the old JSON column
    op.drop_column("images", "detection_result")


def downgrade() -> None:
    op.add_column("images", sa.Column("detection_result", sa.JSON(), nullable=True))
    op.drop_table("detections")
    op.drop_column("images", "threshold")
    op.drop_column("images", "model_version")
    op.drop_column("images", "model_type")
    op.drop_column("images", "content_hash")
    op.drop_column("images", "file_size")
    op.drop_column("images", "height")
    op.drop_column("images", "width")

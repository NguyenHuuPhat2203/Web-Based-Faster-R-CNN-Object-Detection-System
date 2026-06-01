"""Create initial tables (users, token_blacklist, images)

Revision ID: 001
Revises:
Create Date: 2026-06-01
"""
from typing import Union, Sequence
from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table("users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("username", sa.String(100), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=True),
        sa.Column("google_id", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_id", "users", ["id"], unique=False)

    op.create_table("token_blacklist",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("jti", sa.String(255), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_token_blacklist_jti", "token_blacklist", ["jti"], unique=True)
    op.create_index("ix_token_blacklist_id", "token_blacklist", ["id"], unique=False)

    op.create_table("images",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("original_name", sa.String(500), nullable=False),
        sa.Column("stored_name", sa.String(255), nullable=False),
        sa.Column("filepath", sa.String(1000), nullable=False),
        sa.Column("mime_type", sa.String(50), nullable=False),
        sa.Column("detection_result", sa.JSON(), nullable=True),
        sa.Column("uploaded_at", sa.DateTime(), nullable=True),
        sa.Column("deleted", sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_images_id", "images", ["id"], unique=False)


def downgrade() -> None:
    op.drop_table("images")
    op.drop_table("token_blacklist")
    op.drop_table("users")

"""Enforce uniqueness of users.google_id

The ORM declares ``google_id`` unique but migration 001 never created the
constraint, so two accounts could link the same Google identity.

Revision ID: 005
Revises: 004
"""

from collections.abc import Sequence

from alembic import op

revision: str = "005"
down_revision: str | None = "004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CONSTRAINT_NAME = "uq_users_google_id"


def upgrade() -> None:
    with op.batch_alter_table("users") as batch_op:
        batch_op.create_unique_constraint(_CONSTRAINT_NAME, ["google_id"])


def downgrade() -> None:
    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_constraint(_CONSTRAINT_NAME, type_="unique")

"""Add ON DELETE CASCADE to images.user_id foreign key

The ORM declares ``ForeignKey("users.id", ondelete="CASCADE")`` but migration 001
created the constraint without it.

Revision ID: 004
Revises: 003
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "004"
down_revision: str | None = "003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# SQLite reflects the unnamed FK from 001; the convention gives it a stable name.
_NAMING_CONVENTION = {
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
}
_COMPOSITE_INDEX = "ix_images_user_id_deleted_uploaded_at"


def _user_fk_name() -> str:
    """Return the current name of the images.user_id FK (convention name if unnamed)."""
    foreign_keys = sa.inspect(op.get_bind()).get_foreign_keys("images")
    for foreign_key in foreign_keys:
        if foreign_key["constrained_columns"] == ["user_id"]:
            return foreign_key["name"] or _NAMING_CONVENTION["fk"] % {
                "table_name": "images",
                "column_0_name": "user_id",
                "referred_table_name": "users",
            }
    raise RuntimeError("images.user_id foreign key not found")


def _replace_user_fk(ondelete: str | None) -> None:
    """Recreate the images.user_id FK with the given ON DELETE behavior."""
    old_name = _user_fk_name()
    # Batch table copy (SQLite) cannot reflect the expression index; rebuild it.
    op.drop_index(_COMPOSITE_INDEX, table_name="images")
    with op.batch_alter_table(
        "images", naming_convention=_NAMING_CONVENTION
    ) as batch_op:
        batch_op.drop_constraint(old_name, type_="foreignkey")
        batch_op.create_foreign_key(
            "fk_images_user_id_users",
            "users",
            ["user_id"],
            ["id"],
            ondelete=ondelete,
        )
    op.create_index(
        _COMPOSITE_INDEX,
        "images",
        ["user_id", "deleted", sa.text("uploaded_at DESC")],
    )


def upgrade() -> None:
    _replace_user_fk("CASCADE")


def downgrade() -> None:
    _replace_user_fk(None)

"""Alembic environment: reuses the application's settings and ORM metadata.

Run from the repo root: ``alembic -c backend/alembic.ini upgrade head``
(``prepend_sys_path = .`` in alembic.ini puts the repo root on ``sys.path``).
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

import backend.models  # noqa: F401  (registers all models on Base.metadata)
from backend.config import get_settings
from backend.core.database import Base

config = context.config

if config.config_file_name is not None:
    # Keep the application's own loggers alive when run programmatically.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata

# Same URL resolution as the app (relative SQLite is anchored to backend/).
# configparser treats "%" as interpolation, so escape it (e.g. URL-encoded passwords).
config.set_main_option("sqlalchemy.url", get_settings().database_url.replace("%", "%%"))


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without a DB connection)."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            # SQLite cannot ALTER most constraints; autogenerate batch ops for it.
            render_as_batch=connection.dialect.name == "sqlite",
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

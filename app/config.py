"""Configuration helpers for the shopping assistant backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present to simplify local development.
load_dotenv()


def _require_env(key: str, default: str | None = None) -> str:
    """Return the environment variable value or raise a helpful error."""

    value = os.getenv(key, default)
    if value is None:
        raise RuntimeError(f"Environment variable '{key}' must be set.")
    return value


@dataclass(frozen=True)
class Settings:
    """Holds configuration derived from environment variables."""

    postgres_user: str
    postgres_password: str
    postgres_host: str
    postgres_port: int
    postgres_db: str
    data_directory: Path
    data_archive_url: str
    data_archive_name: str
    import_marker_name: str

    @property
    def async_database_url(self) -> str:
        """Return the async SQLAlchemy URL for application usage."""

        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        """Return the sync SQLAlchemy URL for running Alembic migrations."""

        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def data_archive_path(self) -> Path:
        """Compute the on-disk path for the downloaded archive."""

        return self.data_directory / self.data_archive_name

    @property
    def import_marker_path(self) -> Path:
        """Compute the path of the marker file signalling data load completion."""

        return self.data_directory / self.import_marker_name


def _default_postgres_host() -> str:
    """Return the preferred PostgreSQL host for the current environment."""

    return os.getenv("POSTGRES_LOCAL_HOST") or _require_env("POSTGRES_HOST")


def get_settings() -> Settings:
    """Create settings populated from the environment."""

    data_dir_raw = os.getenv("TOROB_DATA_DIR", "/app/data/torob")
    data_dir = Path(data_dir_raw).expanduser().resolve()

    return Settings(
        postgres_user=_require_env("POSTGRES_USER"),
        postgres_password=_require_env("POSTGRES_PASSWORD"),
        postgres_host=_default_postgres_host(),
        postgres_port=int(_require_env("POSTGRES_PORT", "5432")),
        postgres_db=_require_env("POSTGRES_DB"),
        data_directory=data_dir,
        data_archive_url=os.getenv(
            "TOROB_DATA_ARCHIVE_URL",
            "https://drive.google.com/uc?export=download&id=1W4mSI33IbeKkWztK3XmE05F7m4tNYDYu",
        ),
        data_archive_name=os.getenv("TOROB_DATA_ARCHIVE_NAME", "torob-dataset.tar.gz"),
        import_marker_name=os.getenv("TOROB_DATA_IMPORT_MARKER", ".import-complete"),
    )


settings = get_settings()

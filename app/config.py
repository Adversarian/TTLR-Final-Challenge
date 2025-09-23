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
    load_chunk_size: int
    search_similarity_threshold: float
    request_log_directory: Path
    request_log_idle_seconds: int

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

    return _require_env("POSTGRES_HOST")


def _int_from_env(key: str, default: int) -> int:
    """Parse a positive integer from the environment."""

    raw_value = os.getenv(key)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:  # pragma: no cover - defensive programming
        raise RuntimeError(f"Environment variable '{key}' must be an integer") from exc

    if value <= 0:
        raise RuntimeError(f"Environment variable '{key}' must be greater than zero")

    return value


def _float_from_env(key: str, default: float) -> float:
    """Parse a floating-point value in the inclusive range ``[0.0, 1.0]``."""

    raw_value = os.getenv(key)
    if raw_value is None:
        return default

    try:
        value = float(raw_value)
    except ValueError as exc:  # pragma: no cover - defensive programming
        raise RuntimeError(
            f"Environment variable '{key}' must be a floating-point number"
        ) from exc

    if not 0.0 <= value <= 1.0:
        raise RuntimeError(f"Environment variable '{key}' must be between 0.0 and 1.0")

    return value


def _positive_float_from_env(key: str, default: float) -> float:
    """Parse a strictly positive floating-point value from the environment."""

    raw_value = os.getenv(key)
    if raw_value is None:
        return default

    try:
        value = float(raw_value)
    except ValueError as exc:  # pragma: no cover - defensive programming
        raise RuntimeError(
            f"Environment variable '{key}' must be a floating-point number"
        ) from exc

    if value <= 0:
        raise RuntimeError(f"Environment variable '{key}' must be greater than zero")

    return value


def get_settings() -> Settings:
    """Create settings populated from the environment."""

    data_dir_raw = os.getenv("TOROB_DATA_DIR", "/app/data/torob")
    data_dir = Path(data_dir_raw).expanduser().resolve()

    log_dir_raw = os.getenv("TOROB_REQUEST_LOG_DIR")
    if log_dir_raw is None:
        request_log_dir = (data_dir / "logs").resolve()
    else:
        request_log_dir = Path(log_dir_raw).expanduser().resolve()

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
        load_chunk_size=_int_from_env("TOROB_LOAD_CHUNK_SIZE", 10_000),
        search_similarity_threshold=_float_from_env(
            "TOROB_SEARCH_SIMILARITY_THRESHOLD", 0.3
        ),
        request_log_directory=request_log_dir,
        request_log_idle_seconds=_int_from_env("TOROB_REQUEST_LOG_IDLE_SECONDS", 60),
    )


settings = get_settings()


CACHE_TTL_SECONDS = _positive_float_from_env("CACHE_TTL_SECONDS", 120.0)


__all__ = ["settings", "CACHE_TTL_SECONDS"]

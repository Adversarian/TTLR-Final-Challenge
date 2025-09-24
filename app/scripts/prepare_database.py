"""Prepare the PostgreSQL database for local or production deployments."""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

import gdown
from alembic import command
from alembic.config import Config

from ..config import settings
from ..data_loader import TABLE_LOADERS, load_all_tables_sync

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"

LOGGER = logging.getLogger(__name__)


def ensure_data_directory() -> Path:
    """Ensure the dataset directory exists and return it."""

    settings.data_directory.mkdir(parents=True, exist_ok=True)
    return settings.data_directory


def download_archive() -> Path:
    """Download the dataset archive if it does not already exist."""

    archive_path = settings.data_archive_path
    if archive_path.exists():
        LOGGER.info("Archive already present at %s", archive_path)
        return archive_path

    LOGGER.info("Downloading dataset archive from %s", settings.data_archive_url)
    gdown.download(settings.data_archive_url, str(archive_path), quiet=False)
    return archive_path


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    """Safely extract an archive, preventing path traversal."""

    destination = destination.resolve()
    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()
        if not str(member_path).startswith(str(destination)):
            raise RuntimeError(f"Unsafe path detected when extracting: {member.name}")
    archive.extractall(path=destination)


EXPECTED_PARQUET_FILES = [details[0] for details in TABLE_LOADERS.values()]


def extract_archive(archive_path: Path, destination: Path) -> None:
    """Extract the dataset archive if the parquet files are missing."""

    if all((destination / filename).exists() for filename in EXPECTED_PARQUET_FILES):
        LOGGER.info("Parquet files already extracted to %s", destination)
        return

    LOGGER.info("Extracting %s to %s", archive_path, destination)
    with tarfile.open(archive_path, "r:gz") as archive:
        _safe_extract(archive, destination)


def ensure_extensions() -> None:
    """Create required PostgreSQL extensions if they are missing."""
    engine: Engine = create_engine(
        settings.sync_database_url,
        isolation_level="AUTOCOMMIT",
        pool_pre_ping=True,
    )
    try:
        with engine.connect() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
    finally:
        engine.dispose()


def run_migrations() -> None:
    """Run Alembic migrations to ensure the schema is up to date."""

    config = Config(str(ALEMBIC_INI))
    config.set_main_option("sqlalchemy.url", settings.sync_database_url)
    command.upgrade(config, "head")
    logging.getLogger().setLevel(logging.INFO)


def load_dataset() -> None:
    """Load parquet tables into the configured PostgreSQL database."""

    LOGGER.info("Loading parquet tables into PostgreSQL")
    results = load_all_tables_sync(
        settings.data_directory, chunk_size=settings.load_chunk_size
    )
    for table, inserted in results.items():
        LOGGER.info("Inserted %s rows into %s", inserted, table)


def main() -> None:
    """Entry point for preparing the database."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ensure_data_directory()
    ensure_extensions()
    run_migrations()

    if settings.import_marker_path.exists():
        LOGGER.info(
            "Import marker %s found, skipping dataset load", settings.import_marker_path
        )
        return

    archive_path = download_archive()
    extract_archive(archive_path, settings.data_directory)
    load_dataset()
    settings.import_marker_path.touch()
    LOGGER.info("Dataset import completed")


if __name__ == "__main__":
    main()

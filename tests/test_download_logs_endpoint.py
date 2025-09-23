"""Tests for the `/download_logs` endpoint behaviour."""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("POSTGRES_USER", "postgres")
os.environ.setdefault("POSTGRES_PASSWORD", "postgres")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "torob")

import app.main as app_main
import app.logging_utils.judge_requests as request_logging
from app.main import app


class _StubRequestLogger:
    """Minimal logger stub exposing the attributes used by the endpoint."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self.closed = False

    @property
    def directory(self) -> Path:
        return self._directory

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def stub_logger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _StubRequestLogger:
    """Provide a stub request logger backed by a temporary directory."""

    stub = _StubRequestLogger(tmp_path)
    monkeypatch.setattr(app_main, "request_logger", stub)
    monkeypatch.setattr(request_logging, "request_logger", stub)
    return stub


def test_download_logs_returns_404_when_no_logs(
    stub_logger: _StubRequestLogger,
) -> None:
    """If no log files exist the endpoint should return a 404 error."""

    client = TestClient(app)
    try:
        response = client.get("/download_logs")
    finally:
        client.close()

    assert response.status_code == 404
    assert response.json()["detail"] == "No judge request logs available."
    assert stub_logger.closed is True


def test_download_logs_returns_latest_archive(stub_logger: _StubRequestLogger) -> None:
    """The default behaviour should archive only the most recent log file."""

    first = stub_logger.directory / "judge-requests-20240101T000000_000000Z.json"
    second = stub_logger.directory / "judge-requests-20240102T010000_000000Z.json"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    client = TestClient(app)
    try:
        response = client.get("/download_logs")
    finally:
        client.close()

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    disposition = response.headers["content-disposition"]
    assert second.stem in disposition

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    try:
        names = archive.namelist()
        assert names == [second.name]
        assert archive.read(second.name).decode("utf-8") == "second"
    finally:
        archive.close()

    assert stub_logger.closed is True


def test_download_logs_can_include_all_files(stub_logger: _StubRequestLogger) -> None:
    """Setting the `all` query parameter should include every available log."""

    files = [
        stub_logger.directory / "judge-requests-20240101T000000_000000Z.json",
        stub_logger.directory / "judge-requests-20240102T010000_000000Z.json",
    ]
    for index, path in enumerate(files, start=1):
        path.write_text(f"payload-{index}", encoding="utf-8")

    client = TestClient(app)
    try:
        response = client.get("/download_logs", params={"all": "true"})
    finally:
        client.close()

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    disposition = response.headers["content-disposition"]
    assert "all" in disposition

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    try:
        names = sorted(archive.namelist())
        assert names == sorted(path.name for path in files)
        for path in files:
            assert archive.read(path.name).decode("utf-8") == (
                f"payload-{files.index(path) + 1}"
            )
    finally:
        archive.close()

    assert stub_logger.closed is True

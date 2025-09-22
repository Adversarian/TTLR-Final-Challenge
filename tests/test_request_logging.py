"""Unit tests for the judge request logging helpers."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Literal

from pydantic import BaseModel

os.environ.setdefault("POSTGRES_USER", "postgres")
os.environ.setdefault("POSTGRES_PASSWORD", "postgres")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "torob")

from app.logging_utils import RequestLogger  # noqa: E402


class _DummyMessage(BaseModel):
    """Minimal message schema mirroring the production payload."""

    type: Literal["text"]
    content: str


class _DummyRequest(BaseModel):
    """Simplified chat request model for exercising the logger."""

    chat_id: str
    messages: list[_DummyMessage]


def test_logger_groups_entries_and_persists_on_close(tmp_path) -> None:
    """Multiple requests should be grouped by their chat identifier."""

    logger = RequestLogger(directory=tmp_path, inactivity_seconds=30)

    async def _exercise() -> None:
        await logger.log_chat_request(
            _DummyRequest(
                chat_id="test-run",
                messages=[_DummyMessage(type="text", content="first")],
            )
        )
        await logger.log_chat_request(
            _DummyRequest(
                chat_id="test-run",
                messages=[_DummyMessage(type="text", content="second")],
            )
        )
        await logger.log_chat_request(
            _DummyRequest(
                chat_id="topic-other",
                messages=[_DummyMessage(type="text", content="third")],
            )
        )
        await logger.aclose()

    asyncio.run(_exercise())

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1

    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert set(data["requests"].keys()) == {"test-run", "topic-other"}
    assert len(data["requests"]["test-run"]) == 2

    first_entry = data["requests"]["test-run"][0]
    assert first_entry["messages"][0]["content"] == "first"
    assert "received_at" in first_entry
    assert data["started_at"] <= data["ended_at"]


def test_logger_ignores_non_matching_chat_ids(tmp_path) -> None:
    """Requests without the tracked prefix should be ignored entirely."""

    logger = RequestLogger(directory=tmp_path, inactivity_seconds=30)

    async def _exercise() -> None:
        await logger.log_chat_request(
            _DummyRequest(
                chat_id="alpha-run",
                messages=[_DummyMessage(type="text", content="first")],
            )
        )
        await logger.aclose()

    asyncio.run(_exercise())

    assert list(tmp_path.glob("*.json")) == []


def test_logger_closes_after_inactivity(tmp_path) -> None:
    """The logger should automatically persist after a period of inactivity."""

    logger = RequestLogger(directory=tmp_path, inactivity_seconds=0.1)

    async def _exercise() -> None:
        await logger.log_chat_request(
            _DummyRequest(
                chat_id="test-auto",
                messages=[_DummyMessage(type="text", content="payload")],
            )
        )
        await asyncio.sleep(0.3)
        await logger.aclose()

    asyncio.run(_exercise())

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1

    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert list(data["requests"].keys()) == ["test-auto"]

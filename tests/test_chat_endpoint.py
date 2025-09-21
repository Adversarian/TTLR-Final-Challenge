"""Tests for the `/chat` endpoint schema behaviour."""

from __future__ import annotations

from decimal import Decimal
import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("POSTGRES_USER", "postgres")
os.environ.setdefault("POSTGRES_PASSWORD", "postgres")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "torob")

import app.main as app_main
from app.agent import AgentReply
from app.main import app
from app.db import get_session


class _DummySession:
    """Minimal async session stub for dependency overrides in tests."""

    async def execute(self, *args, **kwargs):  # pragma: no cover - not used here
        raise AssertionError("Database should not be queried in this test.")

    async def get(self, *args, **kwargs):  # pragma: no cover - not used here
        return None


async def _session_override():
    """Yield a dummy session without touching a real database."""

    yield _DummySession()


def test_chat_accepts_image_payload() -> None:
    """An image message should be accepted and responded to gracefully."""

    app.dependency_overrides[get_session] = _session_override
    try:
        client = TestClient(app)
        response = client.post(
            "/chat",
            json={
                "chat_id": "image-check",
                "messages": [{"type": "image", "content": "ZmFrZS1pbWFnZS1kYXRh"}],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["message"].startswith("Image messages are not supported yet")
    assert payload["base_random_keys"] is None
    assert payload["member_random_keys"] is None


class _StubAgent:
    """Simple async agent stub returning a prebuilt reply."""

    def __init__(self, reply: AgentReply) -> None:
        self._reply = reply

    async def run(self, *args, **kwargs):  # pragma: no cover - simple passthrough
        return SimpleNamespace(output=self._reply)


def test_numeric_reply_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the agent provides a numeric answer it should replace the message."""

    app.dependency_overrides[get_session] = _session_override
    numeric_reply = AgentReply(
        message="Cheapest price is 120000",
        base_random_keys=["bk-1"],
        numeric_answer=Decimal("120000"),
    )
    monkeypatch.setattr(app_main, "get_agent", lambda: _StubAgent(numeric_reply))

    try:
        client = TestClient(app)
        response = client.post(
            "/chat",
            json={
                "chat_id": "seller-stat",
                "messages": [{"type": "text", "content": "cheapest price?"}],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["message"] == "120000"
    assert payload["base_random_keys"] == ["bk-1"]
    assert payload["member_random_keys"] is None


def test_invalid_numeric_reply_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-finite numeric answers should trigger an internal server error."""

    app.dependency_overrides[get_session] = _session_override
    bad_reply = AgentReply.model_construct(message="NaN", numeric_answer=Decimal("NaN"))
    monkeypatch.setattr(app_main, "get_agent", lambda: _StubAgent(bad_reply))
    monkeypatch.setattr(AgentReply, "clipped", lambda self: self)

    try:
        client = TestClient(app)
        response = client.post(
            "/chat",
            json={
                "chat_id": "seller-stat",
                "messages": [{"type": "text", "content": "cheapest price?"}],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    payload = response.json()
    assert payload["detail"] == "Agent returned a non-finite statistic."

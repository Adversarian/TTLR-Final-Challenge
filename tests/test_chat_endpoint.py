"""Tests for the `/chat` endpoint schema behaviour."""

from __future__ import annotations

import os

from fastapi.testclient import TestClient

os.environ.setdefault("POSTGRES_USER", "postgres")
os.environ.setdefault("POSTGRES_PASSWORD", "postgres")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "torob")

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

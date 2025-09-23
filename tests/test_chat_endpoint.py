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
import app.logging_utils.judge_requests as request_logging
from app.agent import AgentReply
from app.main import app
from app.db import get_session


class _DummySession:
    """Minimal async session stub for dependency overrides in tests."""

    async def execute(self, *args, **kwargs):  # pragma: no cover - not used here
        raise AssertionError("Database should not be queried in this test.")

    async def get(self, *args, **kwargs):  # pragma: no cover - not used here
        return None


class _DummySessionContext:
    """Return a dummy session for async context manager usage."""

    def __init__(self) -> None:
        self._session = _DummySession()

    async def __aenter__(self) -> _DummySession:
        return self._session

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _DummySessionFactory:
    """Callable returning a context manager around a dummy session."""

    def __call__(self) -> _DummySessionContext:
        return _DummySessionContext()


async def _session_override():
    """Yield a dummy session without touching a real database."""

    yield _DummySession()


@pytest.fixture(autouse=True)
def _override_session_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the FastAPI handler uses a stub session factory during tests."""

    monkeypatch.setattr(app_main, "AsyncSessionLocal", _DummySessionFactory())


def test_chat_accepts_image_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """An image message should route to the vision agent and return its reply."""

    app.dependency_overrides[get_session] = _session_override
    reply = AgentReply(message="پتو", base_random_keys=[], member_random_keys=[])
    monkeypatch.setattr(app_main, "get_image_agent", lambda: _StubAgent(reply))

    try:
        client = TestClient(app)
        response = client.post(
            "/chat",
            json={
                "chat_id": "image-check",
                "messages": [
                    {"type": "text", "content": "شیء اصلی در تصویر چیست؟"},
                    {
                        "type": "image",
                        "content": "data:image/png;base64,ZmFrZS1pbWFnZS1kYXRh",
                    },
                ],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "message": "پتو",
        "base_random_keys": None,
        "member_random_keys": None,
    }


def test_image_routing_when_text_is_last(monkeypatch: pytest.MonkeyPatch) -> None:
    """Presence of any image payload should trigger the vision agent."""

    app.dependency_overrides[get_session] = _session_override
    reply = AgentReply(message="گلدان", base_random_keys=[], member_random_keys=[])
    image_called = False
    text_called = False

    def _stub_image_agent() -> _StubAgent:
        nonlocal image_called
        image_called = True
        return _StubAgent(reply)

    def _stub_text_agent() -> _StubAgent:
        nonlocal text_called
        text_called = True
        return _StubAgent(AgentReply(message="ignored"))

    monkeypatch.setattr(app_main, "get_image_agent", _stub_image_agent)
    monkeypatch.setattr(app_main, "get_agent", _stub_text_agent)

    try:
        client = TestClient(app)
        response = client.post(
            "/chat",
            json={
                "chat_id": "image-check",
                "messages": [
                    {
                        "type": "image",
                        "content": "data:image/png;base64,ZmFrZS1pbWFnZS1kYXRh",
                    },
                    {"type": "text", "content": "چه چیزی در تصویر می‌بینی؟"},
                ],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert image_called is True
    assert text_called is False
    payload = response.json()
    assert payload == {
        "message": "گلدان",
        "base_random_keys": None,
        "member_random_keys": None,
    }


def test_invalid_image_payload_returns_400(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed base64 data should raise a client error before hitting the agent."""

    app.dependency_overrides[get_session] = _session_override
    called = False

    def _stub_agent() -> _StubAgent:
        nonlocal called
        called = True
        return _StubAgent(AgentReply(message="ok"))

    monkeypatch.setattr(app_main, "get_image_agent", _stub_agent)

    try:
        client = TestClient(app)
        response = client.post(
            "/chat",
            json={
                "chat_id": "image-check",
                "messages": [
                    {"type": "text", "content": "describe"},
                    {"type": "image", "content": "data:image/png;base64,@@@"},
                ],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400
    assert called is False
    assert response.json()["detail"].startswith(
        "Invalid base64 image data"
    ) or response.json()["detail"].startswith("Malformed")


class _StubAgent:
    """Simple async agent stub returning a prebuilt reply."""

    def __init__(self, reply: AgentReply) -> None:
        self._reply = reply

    async def run(self, *args, **kwargs):  # pragma: no cover - simple passthrough
        return SimpleNamespace(output=self._reply)


class _RecorderLogger:
    """Test helper capturing chat identifiers that trigger logging."""

    def __init__(self) -> None:
        self.request_chat_ids: list[str] = []
        self.responses: list[tuple[str, app_main.ChatResponse]] = []

    async def log_chat_request(self, request):
        self.request_chat_ids.append(request.chat_id)

    async def log_chat_response(self, chat_id, response):
        self.responses.append((chat_id, response))

    async def aclose(self) -> None:  # pragma: no cover - no-op for tests
        return None


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


def test_prefixed_chat_ids_trigger_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """The `/chat` endpoint should log judge requests with the tracked prefix."""

    app.dependency_overrides[get_session] = _session_override
    recorder = _RecorderLogger()
    monkeypatch.setattr(app_main, "request_logger", recorder)
    monkeypatch.setattr(request_logging, "request_logger", recorder)

    client = TestClient(app)
    try:
        response = client.post(
            "/chat",
            json={
                "chat_id": "test-session",
                "messages": [{"type": "text", "content": "ping"}],
            },
        )
    finally:
        client.close()
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert recorder.request_chat_ids == ["test-session"]
    assert [
        (chat_id, resp.message)
        for chat_id, resp in recorder.responses
    ] == [("test-session", "pong")]

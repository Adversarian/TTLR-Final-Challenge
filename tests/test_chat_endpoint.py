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
from app.agent.multiturn import MultiTurnAgentReply, TurnState
from app.agent.router import RouterDecision
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
    monkeypatch.setattr(app_main, "get_conversation_router", lambda: _StubRouter("single_turn"))
    router_store = _StubRouterDecisionStore()
    monkeypatch.setattr(app_main, "get_router_decision_store", lambda: router_store)


def test_router_decision_accepts_plain_label() -> None:
    """Bare-string router outputs should validate without wrapping in JSON."""

    decision = RouterDecision.model_validate("multi_turn")
    assert decision.route == "multi_turn"


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


class _StubMultiTurnAgent:
    """Stubbed multi-turn agent returning a fixed reply."""

    def __init__(self, reply: MultiTurnAgentReply) -> None:
        self._reply = reply

    async def run(self, *args, **kwargs):  # pragma: no cover - simple passthrough
        return SimpleNamespace(output=self._reply)


class _StubRouter:
    """Router stub that always returns the configured route."""

    def __init__(self, route: str) -> None:
        self._route = route

    async def run(self, *args, **kwargs):  # pragma: no cover - simple passthrough
        return SimpleNamespace(output=RouterDecision(route=self._route))


class _StubRouterDecisionStore:
    """Simple cache used to track routing decisions in tests."""

    def __init__(self) -> None:
        self.routes: dict[str, str] = {}
        self.deleted_ids: list[str] = []
        self.get_calls: list[str] = []

    async def get(self, chat_id: str) -> str | None:
        self.get_calls.append(chat_id)
        return self.routes.get(chat_id)

    async def set(self, chat_id: str, route: str) -> None:
        self.routes[chat_id] = route

    async def discard(self, chat_id: str) -> None:
        self.deleted_ids.append(chat_id)
        self.routes.pop(chat_id, None)

    async def reset(self) -> None:  # pragma: no cover - unused helper
        self.routes.clear()
        self.deleted_ids.clear()
        self.get_calls.clear()


class _StubTurnStateStore:
    """Simple in-memory turn state store used in tests."""

    def __init__(self) -> None:
        self._states: dict[str, TurnState] = {}
        self.deleted_ids: list[str] = []

    async def get(self, chat_id: str) -> TurnState | None:
        return self._states.get(chat_id)

    async def set(self, chat_id: str, state: TurnState) -> None:
        self._states[chat_id] = state

    async def discard(self, chat_id: str) -> None:
        self.deleted_ids.append(chat_id)
        self._states.pop(chat_id, None)

    async def reset(self) -> None:  # pragma: no cover - unused helper
        self._states.clear()
        self.deleted_ids.clear()


class _RecorderLogger:
    """Test helper capturing chat identifiers that trigger logging."""

    def __init__(self) -> None:
        self.request_chat_ids: list[str] = []
        self.responses: list[tuple[str, int, object]] = []

    async def log_chat_request(self, request):
        self.request_chat_ids.append(request.chat_id)

    async def log_chat_response(self, chat_id, response, *, status_code):
        self.responses.append((chat_id, status_code, response))

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
        (chat_id, status_code, resp.message)
        for chat_id, status_code, resp in recorder.responses
    ] == [("test-session", 200, "pong")]


def test_logger_failures_do_not_block_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Errors raised by the logger should not prevent responding to the judge."""

    app.dependency_overrides[get_session] = _session_override

    class _FailingLogger:
        def __init__(self) -> None:
            self.calls = 0

        async def log_chat_request(self, request):
            self.calls += 1
            raise RuntimeError("logger unavailable")

        async def log_chat_response(self, chat_id, response, *, status_code):
            self.calls += 1
            raise RuntimeError("logger unavailable")

        async def aclose(self) -> None:  # pragma: no cover - no-op for tests
            return None

    failing_logger = _FailingLogger()
    monkeypatch.setattr(app_main, "request_logger", failing_logger)
    monkeypatch.setattr(request_logging, "request_logger", failing_logger)

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
    assert response.json()["message"] == "pong"
    assert failing_logger.calls >= 1


def test_agent_error_is_logged_with_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed agent executions should record an error response with status code."""

    app.dependency_overrides[get_session] = _session_override
    recorder = _RecorderLogger()
    monkeypatch.setattr(app_main, "request_logger", recorder)
    monkeypatch.setattr(request_logging, "request_logger", recorder)

    class _FailingAgent:
        async def run(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(app_main, "get_agent", lambda: _FailingAgent())

    client = TestClient(app)
    try:
        response = client.post(
            "/chat",
            json={
                "chat_id": "test-session",
                "messages": [{"type": "text", "content": "lookup"}],
            },
        )
    finally:
        client.close()
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert recorder.request_chat_ids == ["test-session"]
    assert recorder.responses[-1][0] == "test-session"
    assert recorder.responses[-1][1] == 500
    payload = recorder.responses[-1][2]
    assert isinstance(payload, dict)
    assert payload["detail"] == "Agent execution failed."


def test_router_cache_prevents_reclassification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cached routing decisions should bypass the router and reuse the branch."""

    app.dependency_overrides[get_session] = _session_override

    router_store = app_main.get_router_decision_store()
    router_store.routes["cached-chat"] = "multi_turn"

    state_store = _StubTurnStateStore()
    reply = MultiTurnAgentReply(
        message="لطفاً اطلاعات بیشتری بدهید.",
        member_random_key=None,
        done=False,
        action="ask",
        updated_state=TurnState(turn=2),
    )

    monkeypatch.setattr(app_main, "get_turn_state_store", lambda: state_store)
    monkeypatch.setattr(app_main, "get_multi_turn_agent", lambda: _StubMultiTurnAgent(reply))

    def _router_should_not_run():  # pragma: no cover - ensures cache is used
        raise AssertionError("Router was invoked despite cached decision")

    monkeypatch.setattr(app_main, "get_conversation_router", _router_should_not_run)
    monkeypatch.setattr(app_main, "get_agent", lambda: _StubAgent(AgentReply(message="nope")))

    client = TestClient(app)
    try:
        response = client.post(
            "/chat",
            json={
                "chat_id": "cached-chat",
                "messages": [{"type": "text", "content": "سلام"}],
            },
        )
    finally:
        client.close()
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["member_random_keys"] is None
    assert payload["message"] == "لطفاً اطلاعات بیشتری بدهید."
    assert router_store.get_calls == ["cached-chat"]
    assert router_store.deleted_ids == []
    assert router_store.routes["cached-chat"] == "multi_turn"


def test_multi_turn_branch_returns_member_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the router selects multi-turn, the specialised agent should handle the turn."""

    app.dependency_overrides[get_session] = _session_override

    router_store = app_main.get_router_decision_store()
    store = _StubTurnStateStore()
    reply = MultiTurnAgentReply(
        message="این گزینه مناسب است.",
        member_random_key="member-123",
        done=True,
        action="return",
        updated_state=TurnState(turn=6),
    )

    monkeypatch.setattr(app_main, "get_turn_state_store", lambda: store)
    monkeypatch.setattr(app_main, "get_multi_turn_agent", lambda: _StubMultiTurnAgent(reply))
    monkeypatch.setattr(app_main, "get_conversation_router", lambda: _StubRouter("multi_turn"))

    fallback_called = False

    def _failing_single_turn_agent() -> _StubAgent:
        nonlocal fallback_called
        fallback_called = True
        return _StubAgent(AgentReply(message="nope"))

    monkeypatch.setattr(app_main, "get_agent", _failing_single_turn_agent)

    client = TestClient(app)
    try:
        response = client.post(
            "/chat",
            json={
                "chat_id": "multi-turn", 
                "messages": [{"type": "text", "content": "سلام"}],
            },
        )
    finally:
        client.close()
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["member_random_keys"] == ["member-123"]
    assert payload["message"] == "این گزینه مناسب است."
    assert fallback_called is False
    assert store.deleted_ids == ["multi-turn"]
    assert router_store.deleted_ids == ["multi-turn"]

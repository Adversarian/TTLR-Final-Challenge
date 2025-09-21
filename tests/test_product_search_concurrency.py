"""Regression tests for concurrent product search tool execution."""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Tuple

import anyio
import pytest

from app.agent import AgentDependencies
from app.agent.tools import _search_base_products


class _RecordingSession:
    """Stub async session that records method usage."""

    def __init__(self, label: str, log: List[Tuple[str, str]]) -> None:
        self._label = label
        self._log = log

    async def get(self, *args, **kwargs):
        self._log.append((self._label, "get"))
        await anyio.sleep(0)
        return None

    async def execute(self, *args, **kwargs):
        self._log.append((self._label, "execute"))
        await anyio.sleep(0)
        return []


class _SessionContext:
    """Async context manager returning a recorded session."""

    def __init__(self, session: _RecordingSession) -> None:
        self._session = session

    async def __aenter__(self) -> _RecordingSession:
        return self._session

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _RecordingSessionFactory:
    """Factory yielding unique recording sessions per invocation."""

    def __init__(self) -> None:
        self.calls = 0
        self.log: List[Tuple[str, str]] = []

    def __call__(self) -> _SessionContext:
        self.calls += 1
        session = _RecordingSession(f"session-{self.calls}", self.log)
        return _SessionContext(session)


@pytest.fixture
def anyio_backend() -> str:
    """Limit anyio-powered tests to the asyncio backend for the suite."""

    return "asyncio"


@pytest.mark.anyio("asyncio")
async def test_search_base_products_allows_parallel_calls() -> None:
    """Concurrent search invocations should acquire independent sessions."""

    factory = _RecordingSessionFactory()
    deps = AgentDependencies(
        session=_RecordingSession("legacy", factory.log),
        session_factory=factory,
    )

    async def _run_search(query: str) -> None:
        ctx = SimpleNamespace(deps=deps)
        result = await _search_base_products(ctx, query)
        assert result.matches == []

    async with anyio.create_task_group() as tg:
        tg.start_soon(_run_search, "first query")
        tg.start_soon(_run_search, "second query")

    assert factory.calls == 2
    assert {label for label, _ in factory.log} == {"session-1", "session-2"}

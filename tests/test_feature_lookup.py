"""Unit tests for the catalogue feature lookup helper."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.agent import AgentDependencies, _fetch_feature_details


class _StubProduct:
    """Simple container holding fake extra feature metadata."""

    def __init__(self, extra_features: dict) -> None:
        self.extra_features = extra_features


class _StubSession:
    """Async session stub returning a canned product."""

    def __init__(self, extra_features: dict) -> None:
        self._product = _StubProduct(extra_features)

    async def get(self, *args, **kwargs):
        return self._product

    async def execute(self, *args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("execute should not be called when get succeeds")


class _StubSessionContext:
    """Context manager returning the provided stub session."""

    def __init__(self, session: _StubSession) -> None:
        self._session = session

    async def __aenter__(self) -> _StubSession:
        return self._session

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _StubSessionFactory:
    """Callable returning an async context manager around a stub session."""

    def __init__(self, session: _StubSession) -> None:
        self._session = session

    def __call__(self) -> _StubSessionContext:
        return _StubSessionContext(self._session)


def test_feature_lookup_returns_complete_map() -> None:
    """The helper should expose every flattened feature/value pair."""

    async def _invoke() -> None:
        feature_blob = {
            "General": {"Color": "Red", "Sizes": ["Small", "Large"]},
            "Weight": "10 kg",
        }
        session = _StubSession(feature_blob)
        session_factory = _StubSessionFactory(session)
        ctx = SimpleNamespace(
            deps=AgentDependencies(session=session, session_factory=session_factory)
        )

        result = await _fetch_feature_details(ctx, " BK-123 ")

        assert result.base_random_key == "BK-123"
        assert [feature.name for feature in result.features] == [
            "General Color",
            "General Sizes",
            "Weight",
        ]
        assert [feature.value for feature in result.features] == [
            "Red",
            "Small, Large",
            "10 kg",
        ]
        assert result.available_features == [
            "General Color",
            "General Sizes",
            "Weight",
        ]

    asyncio.run(_invoke())

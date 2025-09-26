"""Unit tests for the multi-turn tools module."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.agent.multiturn.tools import SearchMembersResult, _search_members


class _RecordingSession:
    """Capture execute calls without touching a real database."""

    def __init__(self) -> None:
        self.calls = []

    async def execute(self, stmt, params):  # pragma: no cover - simple stub
        self.calls.append({"stmt": stmt, "params": params})
        assert params["query_tokens_json"] == ["لوستر سقفی", "اتاق نشیمن"]
        assert params["any_query_text"] == '"لوستر سقفی" OR "اتاق نشیمن"'
        assert params["has_any_query"] is True
        assert params["brand_name_query"] is None
        assert params["category_name_query"] is None
        assert params["city_name_query"] is None
        sql_text = str(stmt)
        assert "0.105 * ts_rank_cd(" in sql_text
        assert "0.195 * ts_rank_cd(" in sql_text
        assert sql_text.index("0.195 * ts_rank_cd") > sql_text.index("0.105 * ts_rank_cd")
        return _StubResult({"count": 0, "topK": [], "distributions": {}})


class _StubResult:
    """Provide the minimal interface consumed by _search_members."""

    def __init__(self, payload):
        self._payload = payload

    def one(self):
        return SimpleNamespace(_mapping={"payload": self._payload})


def test_search_members_accepts_multiple_query_tokens() -> None:
    """Ensure multiple token inputs flow through without binding errors."""

    session = _RecordingSession()
    ctx = SimpleNamespace(deps=SimpleNamespace(session=session))

    result = asyncio.run(
        _search_members(
            ctx,
            query_tokens=["لوستر سقفی", "اتاق نشیمن"],
        )
    )

    assert isinstance(result, SearchMembersResult)
    assert result.count == 0
    assert session.calls, "Expected the stub session to record the execution"

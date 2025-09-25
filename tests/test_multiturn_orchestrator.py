"""Tests for the multi-turn coordinator orchestration layer."""

from __future__ import annotations

import asyncio
from typing import cast

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.dependencies import AgentDependencies
from app.agent.multiturn import orchestrator as orchestrator_module
from app.agent.multiturn.contracts import StopReason
from app.agent.multiturn.orchestrator import MultiTurnCoordinator
from app.agent.multiturn.policy import PolicyTurnResult
from app.agent.schemas import AgentReply


@pytest.fixture
def anyio_backend() -> str:
    """Limit AnyIO tests to the asyncio backend."""

    return "asyncio"


def _deps() -> AgentDependencies:
    """Return stubbed agent dependencies for coordinator tests."""

    return AgentDependencies(
        session=cast(AsyncSession, object()),
        session_factory=cast(async_sessionmaker[AsyncSession], object()),
    )


@pytest.mark.anyio("asyncio")
async def test_coordinator_replays_completed_selection(monkeypatch) -> None:
    coordinator = MultiTurnCoordinator()
    calls = 0

    async def _complete(**_: object) -> PolicyTurnResult:
        nonlocal calls
        calls += 1
        return PolicyTurnResult(
            reply=AgentReply(message="ok", member_random_keys=["m-1"]),
            stop_reason=StopReason.FOUND_UNIQUE_MEMBER,
        )

    monkeypatch.setattr(orchestrator_module, "execute_policy_turn", _complete)

    first = await coordinator.run(user_prompt="۲", deps=_deps(), chat_id="chat-1")
    assert first.output.member_random_keys == ["m-1"]
    assert calls == 1

    async def _should_not_run(**_: object) -> PolicyTurnResult:
        raise AssertionError("execute_policy_turn should not run for repeated selections")

    monkeypatch.setattr(orchestrator_module, "execute_policy_turn", _should_not_run)

    second = await coordinator.run(user_prompt="۲", deps=_deps(), chat_id="chat-1")
    assert second.output.member_random_keys == ["m-1"]
    assert calls == 1


@pytest.mark.anyio("asyncio")
async def test_coordinator_timeout_returns_fallback(monkeypatch) -> None:
    coordinator = MultiTurnCoordinator()
    original_timeout = orchestrator_module._TURN_TIMEOUT_SECONDS

    async def _slow_execute(**_: object) -> PolicyTurnResult:
        await asyncio.sleep(0.05)
        return PolicyTurnResult(
            reply=AgentReply(message="ok", member_random_keys=["m-timeout"]),
            stop_reason=StopReason.FOUND_UNIQUE_MEMBER,
        )

    monkeypatch.setattr(orchestrator_module, "execute_policy_turn", _slow_execute)
    monkeypatch.setattr(orchestrator_module, "_TURN_TIMEOUT_SECONDS", 0.01)

    fallback = await coordinator.run(user_prompt="سلام", deps=_deps(), chat_id="chat-timeout")
    assert "زمان پاسخ‌گویی" in (fallback.output.message or "")
    assert not fallback.output.member_random_keys

    async def _fast_execute(**_: object) -> PolicyTurnResult:
        return PolicyTurnResult(
            reply=AgentReply(message="done", member_random_keys=["m-ok"]),
            stop_reason=StopReason.FOUND_UNIQUE_MEMBER,
        )

    monkeypatch.setattr(orchestrator_module, "execute_policy_turn", _fast_execute)
    monkeypatch.setattr(orchestrator_module, "_TURN_TIMEOUT_SECONDS", original_timeout)

    result = await coordinator.run(user_prompt="۲", deps=_deps(), chat_id="chat-timeout")
    assert result.output.member_random_keys == ["m-ok"]

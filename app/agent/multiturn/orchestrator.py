"""Coordinator that applies the dialogue policy across turns."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import monotonic
from types import SimpleNamespace
from typing import Sequence

import logfire
from pydantic_ai.usage import UsageLimits

from ..dependencies import AgentDependencies
from ..logging import _ensure_logfire
from ..schemas import AgentReply
from .contracts import StopReason, TurnState
from .memory import ConversationMemory
from .nlu import normalize_text
from .policy import PolicyTurnResult, execute_policy_turn


_TURN_TIMEOUT_SECONDS = 25.0
_COMPLETED_TTL_SECONDS = 60.0


@dataclass
class _CompletedTurn:
    """Cache of the most recent completed selection for idempotency."""

    reply: AgentReply
    normalized_message: str
    timestamp: float

    def matches(self, normalized_message: str) -> bool:
        return bool(self.normalized_message) and self.normalized_message == normalized_message

    def is_expired(self, now: float) -> bool:
        return now - self.timestamp > _COMPLETED_TTL_SECONDS


class _RunResult(SimpleNamespace):
    """Result wrapper mimicking the Pydantic-AI agent response."""

    def all_messages(self) -> Sequence[str]:  # pragma: no cover - compatibility shim
        return []

    def new_messages(self) -> Sequence[str]:  # pragma: no cover - compatibility shim
        return []


class MultiTurnCoordinator:
    """Stateful orchestrator that runs the multi-turn policy for each chat."""

    def __init__(self, memory: ConversationMemory | None = None) -> None:
        _ensure_logfire()
        self._memory = memory or ConversationMemory()
        self._locks: dict[str, asyncio.Lock] = {}
        self._completed: dict[str, _CompletedTurn] = {}

    async def run(  # noqa: D401 - mirrors Pydantic-AI signature
        self,
        *,
        user_prompt: str,
        deps: AgentDependencies,
        chat_id: str,
        usage_limits: UsageLimits | None = None,  # pragma: no cover - accepted but unused
    ) -> _RunResult:
        """Execute one multi-turn step and persist the updated conversation state."""

        del usage_limits  # Multi-turn policy does not rely on token limits directly.

        lock = self._get_lock(chat_id)
        async with lock:
            cached = self._consume_completed_reply(chat_id, user_prompt)
            if cached is not None:
                return _RunResult(output=cached)

            state = self._restore_state(chat_id)
            state_snapshot = state.model_copy(deep=True)
            logfire.info(
                "multi_turn.start_turn",
                chat_id=chat_id,
                turn=state.turn_index,
            )

            try:
                result = await asyncio.wait_for(
                    execute_policy_turn(
                        session=deps.session,
                        state=state,
                        user_message=user_prompt,
                    ),
                    timeout=_TURN_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logfire.warning("multi_turn.timeout", chat_id=chat_id)
                self._restore_snapshot(chat_id, state_snapshot)
                return _RunResult(output=_build_timeout_reply())
            except Exception as exc:  # pragma: no cover - defensive logging path
                logfire.exception("multi_turn.error", chat_id=chat_id, error=str(exc))
                self._restore_snapshot(chat_id, state_snapshot)
                return _RunResult(output=_build_error_reply())

            self._persist_state(chat_id, state, result, user_prompt)

            return _RunResult(output=result.reply)

    def _get_lock(self, chat_id: str) -> asyncio.Lock:
        lock = self._locks.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[chat_id] = lock
        return lock

    def _consume_completed_reply(self, chat_id: str, user_prompt: str) -> AgentReply | None:
        record = self._completed.get(chat_id)
        if not record:
            return None

        now = monotonic()
        if record.is_expired(now):
            self._completed.pop(chat_id, None)
            return None

        normalized = _normalize_user_message(user_prompt)
        if record.matches(normalized):
            logfire.info("multi_turn.repeat_selection", chat_id=chat_id)
            return record.reply

        self._completed.pop(chat_id, None)

        return None

    def _restore_state(self, chat_id: str) -> TurnState:
        record = self._memory.recall(chat_id)
        if record is not None:
            return record.state
        return TurnState()

    def _restore_snapshot(self, chat_id: str, snapshot: TurnState) -> None:
        self._memory.remember(chat_id, snapshot, summary=snapshot.summary)

    def _persist_state(
        self,
        chat_id: str,
        state: TurnState,
        result: PolicyTurnResult,
        user_prompt: str,
    ) -> None:
        stop_reason = result.stop_reason or state.stop_reason
        summary = result.summary or state.summary
        agent_messages: list[str] | None = None
        if result.reply.message:
            agent_messages = [result.reply.message]

        if stop_reason in {StopReason.FOUND_UNIQUE_MEMBER, StopReason.MAX_TURNS_REACHED}:
            self._memory.forget(chat_id)
            self._remember_completion(chat_id, result.reply, user_prompt)
            logfire.info(
                "multi_turn.finish",
                chat_id=chat_id,
                stop_reason=stop_reason.value if stop_reason else None,
            )
            return

        self._memory.remember(chat_id, state, summary=summary, agent_messages=agent_messages)

    def _remember_completion(
        self,
        chat_id: str,
        reply: AgentReply,
        user_prompt: str,
    ) -> None:
        normalized = _normalize_user_message(user_prompt)
        if not reply.member_random_keys or not normalized.isdigit():
            self._completed.pop(chat_id, None)
            return

        self._completed[chat_id] = _CompletedTurn(
            reply=reply,
            normalized_message=normalized,
            timestamp=monotonic(),
        )


def _normalize_user_message(message: str) -> str:
    if not message:
        return ""
    normalized = normalize_text(message)
    return normalized


def _build_timeout_reply() -> AgentReply:
    return AgentReply(
        message="زمان پاسخ‌گویی طولانی شد؛ لطفاً دوباره تلاش کنید یا توضیح کوتاهی بدهید.",
    )


def _build_error_reply() -> AgentReply:
    return AgentReply(
        message="در حال حاضر خطایی رخ داد و نتوانستم ادامه دهم. لطفاً دوباره تلاش کنید.",
    )


__all__ = ["MultiTurnCoordinator"]


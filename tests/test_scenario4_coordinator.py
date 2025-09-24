from decimal import Decimal

import pytest

from app.agent.dependencies import AgentDependencies
from app.agent.multi_turn.coordinator import Scenario4Coordinator
from app.agent.multi_turn.schemas import (
    ClarificationPlan,
    ConstraintExtraction,
    MemberFilterResponse,
    MemberOffer,
    ResolutionSummary,
)
from app.agent.multi_turn.state import Scenario4ConversationState


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _StubSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, *args, **kwargs):  # pragma: no cover - no DB access expected
        raise AssertionError("Database should not be touched in this test")


class _StubSessionFactory:
    def __call__(self):
        return self

    async def __aenter__(self) -> _StubSession:
        return _StubSession()

    async def __aexit__(self, exc_type, exc, tb):
        return False


async def test_forces_final_member_on_last_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    coordinator = Scenario4Coordinator()
    state = await coordinator._store.get("turn-budget")  # type: ignore[attr-defined]
    state.turn_count = 4
    state.locked_base_key = "base-1"

    offer = MemberOffer(
        member_random_key="member-1",
        shop_id=101,
        price=250000,
        has_warranty=True,
        shop_score=Decimal("4.7"),
        city_name="تهران",
        matched_constraints=["دارای گارانتی مطابق درخواست"],
        match_score=0.85,
    )
    state.candidate_offers = [offer]

    extraction = ConstraintExtraction(summary="needs a kettle")
    plan = ClarificationPlan(action="ask_question", question="برند خاصی مد نظر دارید؟", rationale="turn budget exhausted")
    member_response = MemberFilterResponse(offers=[offer])
    final_summary = ResolutionSummary(message="member selected", member_random_key="member-1")

    async def _fake_run_agent(self, *, agent_key, **kwargs):
        if agent_key == "constraint_extractor":
            return extraction
        if agent_key == "clarification":
            return plan
        if agent_key == "member_resolver":
            return member_response
        if agent_key == "finaliser":
            return final_summary
        raise AssertionError(f"Unexpected agent key: {agent_key}")

    monkeypatch.setattr(Scenario4Coordinator, "_run_agent", _fake_run_agent)

    deps = AgentDependencies(session=_StubSession(), session_factory=_StubSessionFactory())
    reply = await coordinator.handle_turn(
        chat_id="turn-budget",
        user_message="خیلی هم ممنون",
        deps=deps,
        usage_limits=None,
    )

    assert reply.member_random_keys == ["member-1"]
    assert reply.message == "member selected"
    assert state.turn_count == state.max_turns
    assert state.completed is True


async def test_force_final_failure_returns_message(monkeypatch: pytest.MonkeyPatch) -> None:
    coordinator = Scenario4Coordinator()
    state = await coordinator._store.get("force-failure")  # type: ignore[attr-defined]
    state.turn_count = 4

    extraction = ConstraintExtraction(summary="needs guidance")
    plan = ClarificationPlan(action="finalize", question=None, rationale="turn cap reached")
    member_response = MemberFilterResponse(offers=[])

    async def _fake_run_agent(self, *, agent_key, **kwargs):
        if agent_key == "constraint_extractor":
            return extraction
        if agent_key == "clarification":
            return plan
        if agent_key == "member_resolver":
            return member_response
        raise AssertionError(f"Unexpected agent key: {agent_key}")

    async def _fake_fallback_member_offer(self, state, deps):
        return None, []

    monkeypatch.setattr(Scenario4Coordinator, "_run_agent", _fake_run_agent)
    monkeypatch.setattr(Scenario4Coordinator, "_fallback_member_offer", _fake_fallback_member_offer)

    deps = AgentDependencies(session=_StubSession(), session_factory=_StubSessionFactory())
    reply = await coordinator.handle_turn(
        chat_id="force-failure",
        user_message="باشه",
        deps=deps,
        usage_limits=None,
    )

    assert reply.member_random_keys == []
    assert reply.message and "نتوانستم" in reply.message
    assert state.completed is True
    assert state.turn_count == state.max_turns


def test_fallback_question_ignores_dismissed_aspects() -> None:
    coordinator = Scenario4Coordinator()
    state = Scenario4ConversationState(chat_id="dismissed")
    state.constraints.category_hint = "یخچال"
    state.constraints.price_min = 1000000
    state.constraints.price_max = 2000000
    state.constraints.city_preferences.add("تهران")
    state.constraints.dismissed_aspects.update({"brand", "price", "city", "warranty", "features"})

    question = coordinator._fallback_question(state)

    assert "برند" not in question
    assert question.endswith("تا سریع‌تر جمع‌بندی کنم.")


class _DummyResult:
    def __init__(self, output, messages):
        self.output = output
        self._messages = messages

    def all_messages(self, *, output_tool_return_content=None):
        return list(self._messages)

    def new_messages(self, *, output_tool_return_content=None):  # pragma: no cover - legacy compatibility
        return [self._messages[-1]] if self._messages else []


class _DummyAgent:
    def __init__(self, transcripts):
        self._transcripts = transcripts
        self.calls = 0
        self.seen_histories = []

    async def run(self, *, message_history=None, **kwargs):
        self.seen_histories.append(message_history)
        if self.calls >= len(self._transcripts):
            raise AssertionError("Too many agent invocations")
        output, messages = self._transcripts[self.calls]
        self.calls += 1
        return _DummyResult(output, messages)


async def test_run_agent_persists_full_history() -> None:
    coordinator = Scenario4Coordinator()
    state = Scenario4ConversationState(chat_id="history")
    deps = AgentDependencies(session=_StubSession(), session_factory=_StubSessionFactory())

    messages_run1 = ["assistant-tool-call", "tool-response"]
    messages_run2 = messages_run1 + ["assistant-final"]
    agent = _DummyAgent([
        ("first-output", messages_run1),
        ("second-output", messages_run2),
    ])

    result1 = await coordinator._run_agent(
        agent_key="probe",
        agent_factory=lambda: agent,
        state=state,
        deps=deps,
        prompt="step one",
        usage_limits=None,
    )

    assert result1 == "first-output"
    assert state.agent_histories["probe"] == messages_run1

    result2 = await coordinator._run_agent(
        agent_key="probe",
        agent_factory=lambda: agent,
        state=state,
        deps=deps,
        prompt="step two",
        usage_limits=None,
    )

    assert result2 == "second-output"
    assert state.agent_histories["probe"] == messages_run2
    assert agent.seen_histories == [None, messages_run1]

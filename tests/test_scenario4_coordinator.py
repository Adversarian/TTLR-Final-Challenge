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

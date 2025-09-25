import pytest

from app.agent.multiturn import MemberDelta, TurnState, execute_policy_turn
from app.agent.multiturn.policy import PolicyTurnResult
from app.agent.multiturn.search import CandidateSearchResult, RankedCandidate


@pytest.fixture
def anyio_backend() -> str:
    """Limit asynchronous tests to the asyncio backend for portability."""

    return "asyncio"


class StubSearch:
    def __init__(self, results: list[CandidateSearchResult]) -> None:
        self._results = list(results)
        self.calls = 0

    async def __call__(self, session, details):  # noqa: D401 - test helper
        self.calls += 1
        if not self._results:
            raise AssertionError("Search stub exhausted")
        return self._results.pop(0)


def _stub_parse(delta: MemberDelta | None = None):
    payload = delta or MemberDelta()

    def _parse(_: str) -> MemberDelta:
        return payload

    return _parse


def _candidate(member_key: str, relevance: float = 0.9) -> RankedCandidate:
    candidate = RankedCandidate(
        member_random_key=member_key,
        base_random_key=f"base-{member_key}",
        product_name=f"محصول {member_key}",
        brand_name="برند",
        city_name="تهران",
        price=750_000,
        shop_score=4.5,
        relevance=relevance,
    )
    candidate.label = (
        f"«{candidate.product_name} — {candidate.brand_name} — {candidate.price:,} تومان — "
        f"فروشنده امتیاز {candidate.shop_score:.1f}»"
    )
    return candidate


@pytest.mark.anyio("asyncio")
async def test_unique_candidate_short_circuits() -> None:
    state = TurnState()
    candidate = _candidate("m-1")
    search = StubSearch([CandidateSearchResult(count=1, candidates=[candidate])])

    result = await execute_policy_turn(
        session=object(),
        state=state,
        user_message="",
        parse_fn=_stub_parse(),
        search_fn=search,
    )

    assert isinstance(result, PolicyTurnResult)
    assert result.reply.member_random_keys == ["m-1"]
    assert "این گزینه دقیقاً" in (result.reply.message or "")
    assert state.stop_reason.name == "FOUND_UNIQUE_MEMBER"
    assert state.turn_index == 2


@pytest.mark.anyio("asyncio")
async def test_present_options_and_select_numeric() -> None:
    state = TurnState()
    candidates = [
        _candidate("m-1", 0.9),
        _candidate("m-2", 0.8),
        _candidate("m-3", 0.7),
    ]
    search = StubSearch([CandidateSearchResult(count=3, candidates=candidates)])

    first = await execute_policy_turn(
        session=object(),
        state=state,
        user_message="",
        parse_fn=_stub_parse(),
        search_fn=search,
    )

    assert state.awaiting_selection is True
    assert "گزینه" in (first.reply.message or "")
    assert state.turn_index == 2

    async def _fail_search(session, details):  # pragma: no cover - should not run
        raise AssertionError("search should not be called during selection")

    second = await execute_policy_turn(
        session=object(),
        state=state,
        user_message="۲",
        parse_fn=_stub_parse(),
        search_fn=_fail_search,
    )

    assert second.reply.member_random_keys == ["m-2"]
    assert state.stop_reason.name == "FOUND_UNIQUE_MEMBER"


@pytest.mark.anyio("asyncio")
async def test_numeric_selection_without_space() -> None:
    state = TurnState()
    candidates = [
        _candidate("m-1", 0.9),
        _candidate("m-2", 0.8),
        _candidate("m-3", 0.7),
    ]
    search = StubSearch([CandidateSearchResult(count=3, candidates=candidates)])

    await execute_policy_turn(
        session=object(),
        state=state,
        user_message="",
        parse_fn=_stub_parse(),
        search_fn=search,
    )

    async def _fail_search(session, details):  # pragma: no cover - should not run
        raise AssertionError("search should not be called during selection")

    result = await execute_policy_turn(
        session=object(),
        state=state,
        user_message="گزینه۲",
        parse_fn=_stub_parse(),
        search_fn=_fail_search,
    )

    assert result.reply.member_random_keys == ["m-2"]
    assert state.stop_reason.name == "FOUND_UNIQUE_MEMBER"


@pytest.mark.anyio("asyncio")
async def test_invalid_selection_does_not_advance_turn() -> None:
    state = TurnState()
    candidates = [
        _candidate("m-1", 0.9),
        _candidate("m-2", 0.8),
    ]
    search = StubSearch([CandidateSearchResult(count=2, candidates=candidates)])

    await execute_policy_turn(
        session=object(),
        state=state,
        user_message="",
        parse_fn=_stub_parse(),
        search_fn=search,
    )

    assert state.turn_index == 2

    async def _fail_search(session, details):  # pragma: no cover - should not run
        raise AssertionError("search should not run when confirming selection")

    reminder = await execute_policy_turn(
        session=object(),
        state=state,
        user_message="۵",  # خارج از بازه موجود
        parse_fn=_stub_parse(),
        search_fn=_fail_search,
    )

    assert state.turn_index == 2
    assert "شماره" in (reminder.reply.message or "")


@pytest.mark.anyio("asyncio")
async def test_relaxation_applies_and_recovers() -> None:
    state = TurnState()
    state.details.keywords = ["ملحفه"]
    state.details.min_price = 1_000_000
    empty = CandidateSearchResult(count=0, candidates=[])
    recovered = CandidateSearchResult(count=1, candidates=[_candidate("m-5")])
    search = StubSearch([empty, recovered])

    result = await execute_policy_turn(
        session=object(),
        state=state,
        user_message="",
        parse_fn=_stub_parse(),
        search_fn=search,
    )

    assert "جست‌وجو" in (result.reply.message or "")
    assert state.details.keywords == []
    assert state.details.min_price == 1_000_000  # price relaxation not needed yet
    assert result.reply.member_random_keys == ["m-5"]


@pytest.mark.anyio("asyncio")
async def test_turn_limit_picks_top_candidate() -> None:
    state = TurnState(turn_index=5)
    candidates = [
        _candidate("m-1", 0.95),
        _candidate("m-2", 0.90),
    ]
    search = StubSearch([CandidateSearchResult(count=2, candidates=candidates)])

    result = await execute_policy_turn(
        session=object(),
        state=state,
        user_message="",
        parse_fn=_stub_parse(),
        search_fn=search,
    )

    assert "۵ نوبت" in (result.reply.message or "")
    assert result.reply.member_random_keys == ["m-1"]
    assert state.stop_reason.name == "MAX_TURNS_REACHED"


@pytest.mark.anyio("asyncio")
async def test_question_sequence_tracks_product_and_shop_scope() -> None:
    state = TurnState()
    abundant = CandidateSearchResult(count=12, candidates=[_candidate("m-1") for _ in range(5)])
    search = StubSearch([abundant, abundant])

    first = await execute_policy_turn(
        session=object(),
        state=state,
        user_message="",
        parse_fn=_stub_parse(),
        search_fn=search,
    )

    assert first.reply.message == (
        "برای محدود کردن نتایج، بفرمایید چه برند، دسته یا ویژگی خاصی از کالا مدنظر شماست؟"
    )
    assert "product_scope" in state.asked_questions

    second = await execute_policy_turn(
        session=object(),
        state=state,
        user_message="",
        parse_fn=_stub_parse(),
        search_fn=search,
    )

    assert "shop_scope" in state.asked_questions
    assert "فروشنده" in (second.reply.message or "")


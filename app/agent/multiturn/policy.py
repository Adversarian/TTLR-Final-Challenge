"""Dialogue policy for the multi-turn shopping assistant."""

from __future__ import annotations

import re
from typing import Awaitable, Callable, Iterable, Sequence

import logfire
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..schemas import AgentReply
from .contracts import CandidatePreview, MemberDelta, MemberDetails, StopReason, TurnState
from .nlu import NLUResult, normalize_text, parse_user_message
from .search import CandidateSearchResult, RankedCandidate, search_candidates


_PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
_SELECTION_PATTERN = re.compile(r"\b(\d{1,2})\b")

_RELAXATION_ORDER: Sequence[str] = (
    "keywords",
    "product_attributes",
    "price",
    "brand",
    "city",
    "category",
    "score",
    "warranty",
)

_RELAXATION_MESSAGES: dict[str, str] = {
    "keywords": "عبارات جست‌وجو",
    "product_attributes": "ویژگی‌های توصیفی",
    "price": "بازه قیمت",
    "brand": "فیلتر برند",
    "city": "فیلتر شهر",
    "category": "دسته‌بندی",
    "score": "حداقل امتیاز فروشنده",
    "warranty": "شرط گارانتی",
}

_QUESTION_TEMPLATES: dict[str, str] = {
    "product_scope": (
        "برای محدود کردن نتایج، بفرمایید چه برند، دسته یا ویژگی خاصی از کالا مدنظر شماست؟"
    ),
    "shop_scope": (
        "از نظر فروشنده یا محدوده قیمت چه توقعی دارید؟ اگر گارانتی یا امتیاز خاصی مهم است بفرمایید."
    ),
}


class PolicyTurnResult(BaseModel):
    """Outcome of processing a single dialogue turn."""

    reply: AgentReply = Field(..., description="Assistant message for the current turn.")
    stop_reason: StopReason | None = Field(
        None, description="Reason for finishing the multi-turn flow when applicable."
    )
    summary: str | None = Field(
        None, description="Compact summary suitable for persistence in conversation memory."
    )
    relaxed_steps: list[str] = Field(
        default_factory=list,
        description="Relaxation steps applied before producing the reply.",
    )


async def execute_policy_turn(
    *,
    session: AsyncSession,
    state: TurnState,
    user_message: str,
    parse_fn: Callable[[str], NLUResult] = parse_user_message,
    search_fn: Callable[[AsyncSession, MemberDetails], Awaitable[CandidateSearchResult]] = search_candidates,
) -> PolicyTurnResult:
    """Resolve a single user turn by applying NLU, search, and question strategy."""

    current_turn = state.turn_index

    with logfire.span("multi_turn.turn", turn=current_turn, awaiting_selection=state.awaiting_selection):
        # Selection handling takes precedence when options were previously shown.
        selection_reply = _handle_option_selection(state, user_message)
        if selection_reply is not None:
            if selection_reply.stop_reason is not None:
                state.advance_turn()
            return selection_reply

        # Apply fresh constraints from the latest utterance.
        nlu_result = parse_fn(user_message)
        delta = nlu_result.delta if isinstance(nlu_result, NLUResult) else nlu_result
        if isinstance(delta, MemberDelta):
            state.apply_delta(delta)
            if delta.summary:
                state.summary = delta.summary
        else:  # pragma: no cover - defensive
            raise TypeError("NLU parser returned an unsupported payload.")

        result = await search_fn(session, state.details)
        state.candidate_count = result.count
        state.candidates = [_preview_from_ranked(candidate) for candidate in result.candidates]

        relaxed_steps: list[str] = []
        if result.count == 0:
            result, relaxed_steps = await _relax_constraints(session, state, search_fn)
            state.candidate_count = result.count
            state.candidates = [
                _preview_from_ranked(candidate) for candidate in result.candidates
            ]

        message_prefix = _compose_relaxation_notice(relaxed_steps)

        if state.candidate_count == 0:
            reply = AgentReply(
                message=(
                    message_prefix
                    + "هیچ گزینه‌ای با معیارهای فعلی پیدا نشد. لطفاً اطلاعات بیشتری بدهید یا محدودیت دیگری را بیان کنید."
                )
            )
            state.stop_reason = StopReason.RELAXATION_FAILED
            state.advance_turn()
            return PolicyTurnResult(
                reply=reply,
                stop_reason=state.stop_reason,
                summary=state.summary,
                relaxed_steps=relaxed_steps,
            )

        if state.candidate_count == 1:
            candidate = result.candidates[0]
            reply = AgentReply(
                message=(
                    message_prefix
                    + "این گزینه دقیقاً با شرایط شما تطابق دارد:\n"
                    + _format_option_line(1, candidate)
                ),
                member_random_keys=[candidate.member_random_key],
            )
            state.stop_reason = StopReason.FOUND_UNIQUE_MEMBER
            state.awaiting_selection = False
            state.advance_turn()
            return PolicyTurnResult(
                reply=reply,
                stop_reason=state.stop_reason,
                summary=state.summary,
                relaxed_steps=relaxed_steps,
            )

        # Multiple candidates remain. Decide whether to force a pick or ask a follow-up.
        if current_turn >= 5:
            candidate = result.candidates[0]
            reply = AgentReply(
                message=(
                    message_prefix
                    + "۵ نوبت گفتگو کامل شد؛ بهترین گزینه موجود را انتخاب کردم:\n"
                    + _format_option_line(1, candidate)
                ),
                member_random_keys=[candidate.member_random_key],
            )
            state.stop_reason = StopReason.MAX_TURNS_REACHED
            state.awaiting_selection = False
            state.advance_turn()
            return PolicyTurnResult(
                reply=reply,
                stop_reason=state.stop_reason,
                summary=state.summary,
                relaxed_steps=relaxed_steps,
            )

        if 2 <= state.candidate_count <= 5 or _questions_exhausted(state):
            reply = AgentReply(
                message=message_prefix
                + _render_options_message(result.candidates),
            )
            state.awaiting_selection = True
            state.advance_turn()
            return PolicyTurnResult(
                reply=reply,
                stop_reason=None,
                summary=state.summary,
                relaxed_steps=relaxed_steps,
            )

        question_key = _pick_question(state)
        if question_key is None:
            reply = AgentReply(
                message=message_prefix
                + _render_options_message(result.candidates),
            )
            state.awaiting_selection = True
            state.advance_turn()
            return PolicyTurnResult(
                reply=reply,
                stop_reason=None,
                summary=state.summary,
                relaxed_steps=relaxed_steps,
            )

        question_text = _QUESTION_TEMPLATES[question_key]
        state.asked_questions.add(question_key)
        state.awaiting_selection = False
        reply = AgentReply(message=message_prefix + question_text)
        state.advance_turn()
        return PolicyTurnResult(
            reply=reply,
            stop_reason=None,
            summary=state.summary,
            relaxed_steps=relaxed_steps,
        )


def _handle_option_selection(state: TurnState, message: str) -> PolicyTurnResult | None:
    """Return a reply when the user selects a numbered option."""

    if not state.awaiting_selection or not state.candidates:
        return None

    normalized = normalize_text(message)
    match = _SELECTION_PATTERN.search(normalized)
    if not match:
        return PolicyTurnResult(
            reply=AgentReply(
                message=(
                    "لطفاً شماره یکی از گزینه‌های پیشنهادی را به صورت عدد وارد کنید (مثلاً ۱ یا 2).\n"
                    + _render_preview_options(state.candidates)
                ),
            ),
            stop_reason=None,
            summary=state.summary,
        )

    choice = int(match.group(1))
    if not 1 <= choice <= len(state.candidates):
        return PolicyTurnResult(
            reply=AgentReply(
                message=(
                    "عدد انتخاب‌شده خارج از محدوده است؛ لطفاً از بین گزینه‌های موجود یک شماره معتبر بگویید.\n"
                    + _render_preview_options(state.candidates)
                ),
            ),
            stop_reason=None,
            summary=state.summary,
        )

    candidate = state.candidates[choice - 1]
    state.stop_reason = StopReason.FOUND_UNIQUE_MEMBER
    state.awaiting_selection = False
    state.candidate_count = 1
    return PolicyTurnResult(
        reply=AgentReply(
            message=(
                "این گزینه را برای شما ثبت کردم:\n"
                + _format_preview_line(choice, candidate)
            ),
            member_random_keys=[candidate.member_random_key],
        ),
        stop_reason=state.stop_reason,
        summary=state.summary,
    )


async def _relax_constraints(
    session: AsyncSession,
    state: TurnState,
    search_fn: Callable[[AsyncSession, MemberDetails], Awaitable[CandidateSearchResult]],
) -> tuple[CandidateSearchResult, list[str]]:
    """Relax constraints following the deterministic order until matches emerge."""

    relaxed: list[str] = []
    for step in _RELAXATION_ORDER:
        if not _apply_relaxation(state, step):
            continue
        relaxed.append(step)
        logfire.info("multi_turn.relax", step=step)
        result = await search_fn(session, state.details)
        if result.count > 0:
            return result, relaxed
    return CandidateSearchResult(), relaxed


def _apply_relaxation(state: TurnState, step: str) -> bool:
    """Apply a single relaxation step; return True if it changed the state."""

    details = state.details
    if step == "keywords" and details.keywords:
        details.keywords.clear()
        return True
    if step == "product_attributes" and details.product_attributes:
        details.product_attributes.clear()
        return True
    if step == "price" and (details.min_price is not None or details.max_price is not None):
        details.min_price = None
        details.max_price = None
        return True
    if step == "brand" and details.brand_names:
        details.brand_names.clear()
        return True
    if step == "city" and details.city_names:
        details.city_names.clear()
        return True
    if step == "category" and details.category_names:
        details.category_names.clear()
        return True
    if step == "score" and details.min_shop_score is not None:
        details.min_shop_score = None
        return True
    if step == "warranty" and details.warranty_required is not None:
        details.warranty_required = None
        return True
    return False


def _compose_relaxation_notice(steps: Iterable[str]) -> str:
    """Return a short Persian prefix describing the applied relaxations."""

    steps = list(steps)
    if not steps:
        return ""
    labels = [_RELAXATION_MESSAGES.get(step, step) for step in steps]
    joined = "، ".join(labels)
    return f"برای ادامه جست‌وجو، {joined} را کمی آزادتر در نظر گرفتم. "


def _render_options_message(candidates: Sequence[RankedCandidate]) -> str:
    """Return a Persian message enumerating candidate options."""

    lines = ["چند گزینه نزدیک پیدا کردم؛ لطفاً شماره مورد دلخواه را بفرمایید:"]
    for index, candidate in enumerate(candidates, start=1):
        lines.append(_format_option_line(index, candidate))
    return "\n".join(lines)


def _render_preview_options(candidates: Sequence[CandidatePreview]) -> str:
    """Return the cached option list formatted for reminders."""

    lines = []
    for index, candidate in enumerate(candidates, start=1):
        lines.append(_format_preview_line(index, candidate))
    return "\n".join(lines)


def _format_option_line(index: int, candidate: RankedCandidate) -> str:
    """Format a ranked candidate as an enumerated option."""

    numeral = str(index).translate(_PERSIAN_DIGITS)
    label = candidate.label or _compose_label(candidate)
    return f"{numeral}) {label}"


def _format_preview_line(index: int, preview: CandidatePreview) -> str:
    """Format a previously cached preview when confirming a selection."""

    numeral = str(index).translate(_PERSIAN_DIGITS)
    brand = preview.brand_name or "بدون برند"
    price = f"{preview.price:,}" if preview.price is not None else "نامشخص"
    score = f"{preview.shop_score:.1f}" if preview.shop_score is not None else "—"
    return (
        f"{numeral}) «{preview.product_name} — {brand} — {price} تومان — "
        f"فروشنده امتیاز {score}»"
    )


def _compose_label(candidate: RankedCandidate) -> str:
    brand = candidate.brand_name or "بدون برند"
    price = f"{candidate.price:,}" if candidate.price is not None else "نامشخص"
    score = f"{candidate.shop_score:.1f}" if candidate.shop_score is not None else "—"
    return (
        f"«{candidate.product_name} — {brand} — {price} تومان — "
        f"فروشنده امتیاز {score}»"
    )


def _preview_from_ranked(candidate: RankedCandidate) -> CandidatePreview:
    """Convert a ranked candidate into a lightweight preview for storage."""

    return CandidatePreview(
        member_random_key=candidate.member_random_key,
        base_random_key=candidate.base_random_key,
        product_name=candidate.product_name,
        brand_name=candidate.brand_name,
        shop_name=None,
        city_name=candidate.city_name,
        price=candidate.price,
        shop_score=candidate.shop_score,
    )


def _pick_question(state: TurnState) -> str | None:
    """Return the next clarification question key, if any."""

    if "product_scope" not in state.asked_questions:
        return "product_scope"
    if "shop_scope" not in state.asked_questions:
        needs_shop_hint = not any(
            (
                state.details.min_price is not None,
                state.details.max_price is not None,
                state.details.min_shop_score is not None,
                state.details.warranty_required is not None,
            )
        )
        if needs_shop_hint:
            return "shop_scope"
    if "shop_scope" not in state.asked_questions:
        return "shop_scope"
    return None


def _questions_exhausted(state: TurnState) -> bool:
    """Return ``True`` when all predefined question prompts were asked."""

    return all(key in state.asked_questions for key in _QUESTION_TEMPLATES)


__all__ = [
    "PolicyTurnResult",
    "execute_policy_turn",
]


"""Dialogue policy for the multi-turn shopping assistant."""

from __future__ import annotations

import re
from typing import Awaitable, Callable, Iterable, Mapping, Sequence

import logfire
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..schemas import AgentReply
from .contracts import CandidatePreview, MemberDelta, MemberDetails, StopReason, TurnState
from .nlu import NLUResult, normalize_text, parse_user_message
from .search import CandidateSearchResult, RankedCandidate, search_candidates


_PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
_SELECTION_PATTERN = re.compile(r"(?<!\d)(\d{1,2})(?!\d)")

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
    state.stop_reason = None

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
            _log_delta(delta)
            state.apply_delta(delta)
            if delta.summary:
                state.summary = delta.summary
        else:  # pragma: no cover - defensive
            raise TypeError("NLU parser returned an unsupported payload.")

        constraint_summary = _summarize_details(state.details)
        with logfire.span(
            "multi_turn.search",
            turn=current_turn,
            constraints=constraint_summary,
        ):
            result = await search_fn(session, state.details)

        initial_count = result.count
        relaxed_steps: list[str] = []
        if result.count == 0:
            result, relaxed_steps = await _relax_constraints(session, state, search_fn)

        state.candidate_count = result.count
        state.candidates = [_preview_from_ranked(candidate) for candidate in result.candidates]

        log_kwargs = {
            "turn": current_turn,
            "initial_count": initial_count,
            "final_count": state.candidate_count,
            "top_returned": len(result.candidates),
        }
        if relaxed_steps:
            log_kwargs["relaxed_steps"] = list(relaxed_steps)
        logfire.info("multi_turn.search_result", **log_kwargs)

        message_prefix = _compose_relaxation_notice(relaxed_steps)

        if state.candidate_count == 0:
            reply = AgentReply(
                message=(
                    message_prefix
                    + "هیچ گزینه‌ای با معیارهای فعلی پیدا نشد. لطفاً اطلاعات بیشتری بدهید یا محدودیت دیگری را بیان کنید."
                )
            )
            state.stop_reason = StopReason.RELAXATION_FAILED
            logfire.info(
                "multi_turn.stop",
                reason=state.stop_reason.value,
                turn=current_turn,
            )
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
            logfire.info(
                "multi_turn.stop",
                reason=state.stop_reason.value,
                turn=current_turn,
                member_key=candidate.member_random_key,
            )
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
            logfire.info(
                "multi_turn.stop",
                reason=state.stop_reason.value,
                turn=current_turn,
                member_key=candidate.member_random_key,
            )
            state.advance_turn()
            return PolicyTurnResult(
                reply=reply,
                stop_reason=state.stop_reason,
                summary=state.summary,
                relaxed_steps=relaxed_steps,
            )

        if 2 <= state.candidate_count <= 5 or _questions_exhausted(state):
            reason = "few_candidates" if state.candidate_count <= 5 else "questions_exhausted"
            logfire.info(
                "multi_turn.present_options",
                turn=current_turn,
                options=len(result.candidates),
                reason=reason,
            )
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
            logfire.info(
                "multi_turn.present_options",
                turn=current_turn,
                options=len(result.candidates),
                reason="question_fallback",
            )
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
        logfire.info("multi_turn.ask_question", turn=current_turn, question=question_key)
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
    logfire.info(
        "multi_turn.selection",
        choice=choice,
        member_key=candidate.member_random_key,
    )
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
            logfire.info(
                "multi_turn.relax_success",
                steps=list(relaxed),
                count=result.count,
            )
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


def _compact_values(values: Iterable[str], *, limit: int = 3) -> list[str]:
    """Return a sorted, deduplicated list trimmed to the requested size."""

    cleaned = sorted({value for value in values if value})
    if limit and len(cleaned) > limit:
        return cleaned[:limit]
    return cleaned


def _compact_sequence(values: Sequence[str], *, limit: int = 5) -> list[str]:
    """Return the leading slice of a sequence, omitting empty entries."""

    trimmed: list[str] = []
    for value in values:
        if not value:
            continue
        trimmed.append(value)
        if limit and len(trimmed) >= limit:
            break
    return trimmed


def _trimmed_mapping(mapping: Mapping[str, str], *, limit: int = 3) -> dict[str, str]:
    """Return at most ``limit`` non-empty key/value pairs from the mapping."""

    result: dict[str, str] = {}
    for index, (key, value) in enumerate(mapping.items()):
        if limit and index >= limit:
            break
        if not key or not value:
            continue
        result[key] = value
    return result


def _describe_delta(delta: MemberDelta) -> dict[str, object]:
    """Build a compact log payload describing the extracted delta."""

    payload: dict[str, object] = {}
    if delta.brand_names:
        payload["brand"] = _compact_values(delta.brand_names)
    if delta.category_names:
        payload["category"] = _compact_values(delta.category_names)
    if delta.city_names:
        payload["city"] = _compact_values(delta.city_names)
    if delta.min_price is not None or delta.max_price is not None:
        payload["price"] = {"min": delta.min_price, "max": delta.max_price}
    if delta.min_shop_score is not None:
        payload["min_shop_score"] = delta.min_shop_score
    if delta.warranty_required is not None:
        payload["warranty_required"] = delta.warranty_required
    if delta.keywords:
        payload["keywords"] = _compact_values(delta.keywords, limit=5)
    if delta.product_attributes:
        payload["attributes"] = _trimmed_mapping(delta.product_attributes)
    if delta.asked_fields:
        payload["asked_fields"] = _compact_values(delta.asked_fields, limit=5)
    if delta.excluded_fields:
        payload["excluded_fields"] = _compact_values(delta.excluded_fields, limit=5)

    drop_flags = [
        name
        for name, flag in (
            ("brand", delta.drop_brand_names),
            ("category", delta.drop_category_names),
            ("city", delta.drop_city_names),
            ("price", delta.drop_price_range),
            ("score", delta.drop_min_shop_score),
            ("warranty", delta.drop_warranty_requirement),
            ("keywords", delta.drop_keywords),
            ("attributes", delta.drop_product_attributes),
        )
        if flag
    ]
    if drop_flags:
        payload["drops"] = drop_flags
    if delta.summary:
        payload["summary"] = delta.summary[:80]

    return payload


def _log_delta(delta: MemberDelta) -> None:
    """Emit a structured Logfire event summarising the parsed delta."""

    payload = _describe_delta(delta)
    if payload:
        logfire.info("multi_turn.delta", **payload)


def _summarize_details(details: MemberDetails) -> dict[str, object]:
    """Return a trimmed description of the active member constraints."""

    payload: dict[str, object] = {}
    if details.brand_names:
        payload["brands"] = _compact_values(details.brand_names)
    if details.category_names:
        payload["categories"] = _compact_values(details.category_names)
    if details.city_names:
        payload["cities"] = _compact_values(details.city_names)
    if details.min_price is not None or details.max_price is not None:
        payload["price"] = {"min": details.min_price, "max": details.max_price}
    if details.min_shop_score is not None:
        payload["min_shop_score"] = details.min_shop_score
    if details.warranty_required is not None:
        payload["warranty_required"] = details.warranty_required
    if details.keywords:
        payload["keywords"] = _compact_sequence(details.keywords, limit=5)
        payload["keyword_count"] = len(details.keywords)
    if details.product_attributes:
        payload["attributes"] = _trimmed_mapping(details.product_attributes)
    if details.excluded_fields:
        payload["excluded_fields"] = _compact_values(details.excluded_fields, limit=5)
    return payload


def _pick_question(state: TurnState) -> str | None:
    """Return the next clarification question key, if any."""

    if "product_scope" not in state.asked_questions:
        return "product_scope"
    if "shop_scope" in state.asked_questions:
        return None

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
    return None


def _questions_exhausted(state: TurnState) -> bool:
    """Return ``True`` when all predefined question prompts were asked."""

    return all(key in state.asked_questions for key in _QUESTION_TEMPLATES)


__all__ = [
    "PolicyTurnResult",
    "execute_policy_turn",
]


"""Coordinator for the multi-turn scenario 4 workflow."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional

from ..dependencies import AgentDependencies
from .extractor import extract_constraints
from .schemas import ConstraintUpdate, MemberCandidate
from .search import search_members
from .state import MemberSearchState, MultiTurnSession

_TURN_LIMIT = 5
_OPTION_LIMIT = 5


@dataclass
class MultiTurnReply:
    """Represents the assistant response for a multi-turn step."""

    message: str
    member_random_keys: Optional[List[str]] = None


class MultiTurnManager:
    """Stateful manager that handles multi-turn interactions per chat_id."""

    def __init__(self) -> None:
        self._sessions: Dict[str, MultiTurnSession] = {}

    def _get_session(self, chat_id: str) -> MultiTurnSession:
        session = self._sessions.get(chat_id)
        if session is None:
            session = MultiTurnSession(chat_id=chat_id)
            self._sessions[chat_id] = session
        return session

    async def handle(
        self,
        chat_id: str,
        deps: AgentDependencies,
        text_segments: List[str],
    ) -> MultiTurnReply:
        session = self._get_session(chat_id)

        if session.state.finalized_member_key:
            return MultiTurnReply(
                message="جستجو پیش‌تر تکمیل شده است.",
                member_random_keys=[session.state.finalized_member_key],
            )

        if len(text_segments) <= session.processed_message_count:
            return MultiTurnReply(
                message="لطفاً اطلاعات بیشتری درباره محصول مورد نظرتان بفرستید تا ادامه دهیم."
            )

        new_messages = text_segments[session.processed_message_count :]
        session.processed_message_count = len(text_segments)
        incoming_text = "\n".join(new_messages).strip()

        if not incoming_text:
            return MultiTurnReply(
                message="پیام واضحی دریافت نشد. لطفاً توضیح دهید چه محصولی مدنظرتان است."
            )

        update = await extract_constraints(session, deps, incoming_text)
        _apply_update(session.state, update)

        if session.state.finalized_member_key:
            session.state.turns_taken += 1
            return MultiTurnReply(
                message="فروشنده مد نظر پیدا شد.",
                member_random_keys=[session.state.finalized_member_key],
            )

        result = await search_members(deps.session, session.state.filters)
        candidates = list(result.candidates)
        session.state.last_candidates = candidates

        if _should_force_fallback(session.state):
            reply = _finalize_with_best_candidate(session.state)
            if reply is not None:
                return reply

        if not candidates:
            session.state.turns_taken += 1
            message = (
                "هنوز گزینه دقیقی پیدا نشد. لطفاً ویژگی‌های شاخص، حدود قیمت یا شهر مورد"
                " نظر را بگویید تا بتوانم دقیق‌تر جستجو کنم."
            )
            return MultiTurnReply(message=message)

        if len(candidates) == 1:
            best = candidates[0]
            session.state.finalized_member_key = best.member_random_key
            session.state.turns_taken += 1
            return MultiTurnReply(
                message=(
                    "این فروشنده دقیق‌ترین تطابق با توضیحات شما است."
                ),
                member_random_keys=[best.member_random_key],
            )

        if len(candidates) <= _OPTION_LIMIT and not _already_presented(
            session.state, candidates
        ):
            message = _present_options(session.state, candidates)
            session.state.turns_taken += 1
            return MultiTurnReply(message=message)

        question = _choose_question(session.state, candidates)
        session.state.turns_taken += 1
        return MultiTurnReply(message=question)


def _apply_update(state: MemberSearchState, update: ConstraintUpdate) -> None:
    filters = state.filters
    filters.add_text_queries(update.text_queries)
    filters.add_feature_hints(update.feature_hints)

    def _unique_extend(target: List[int], values: Iterable[int]) -> None:
        for item in values:
            if item not in target:
                target.append(item)

    _unique_extend(filters.preferred_shop_ids, update.preferred_shop_ids)
    _unique_extend(filters.allowed_shop_ids, update.allowed_shop_ids)

    mapping = {
        "category_id": ("category_id",),
        "brand_id": ("brand_id",),
        "city_id": ("city_id",),
        "min_price": ("min_price",),
        "max_price": ("max_price",),
        "requires_warranty": ("requires_warranty",),
        "min_score": ("min_score",),
        "max_score": ("max_score",),
        "price_range": ("min_price", "max_price"),
        "score": ("min_score", "max_score"),
    }

    for field_name in update.excluded_fields:
        state.excluded_fields.add(field_name)
        attrs = mapping.get(field_name)
        if attrs:
            for attr in attrs:
                setattr(filters, attr, None)

    for field_name in update.clear_fields:
        attrs = mapping.get(field_name)
        if attrs:
            for attr in attrs:
                setattr(filters, attr, None)

    if update.category_id is not None and "category_id" not in state.excluded_fields:
        filters.category_id = update.category_id
    if update.brand_id is not None and "brand_id" not in state.excluded_fields:
        filters.brand_id = update.brand_id
    if update.city_id is not None and "city_id" not in state.excluded_fields:
        filters.city_id = update.city_id

    if update.min_price is not None:
        filters.min_price = update.min_price
    if update.max_price is not None:
        filters.max_price = update.max_price
    if update.requires_warranty is not None:
        filters.requires_warranty = update.requires_warranty
    if update.min_score is not None:
        filters.min_score = update.min_score
    if update.max_score is not None:
        filters.max_score = update.max_score

    if update.selected_member_random_key:
        state.finalized_member_key = update.selected_member_random_key

    if state.pending_question is not None:
        state.asked_questions.add(state.pending_question)
        state.pending_question = None


def _should_force_fallback(state: MemberSearchState) -> bool:
    return state.turns_taken >= (_TURN_LIMIT - 1) and not state.finalized_member_key


def _finalize_with_best_candidate(state: MemberSearchState) -> Optional[MultiTurnReply]:
    if not state.last_candidates:
        return None
    best = max(state.last_candidates, key=lambda item: item.score)
    state.finalized_member_key = best.member_random_key
    state.fallback_used = True
    state.turns_taken += 1
    message = (
        "به دلیل محدودیت نوبت‌ها، نزدیک‌ترین گزینه موجود را انتخاب کردم. اگر "
        "نیاز به تغییر دارید لطفاً اعلام کنید."
    )
    return MultiTurnReply(message=message, member_random_keys=[best.member_random_key])


def _already_presented(state: MemberSearchState, candidates: List[MemberCandidate]) -> bool:
    keys = {candidate.member_random_key for candidate in candidates[:_OPTION_LIMIT]}
    return keys.issubset(state.candidates_shown)


def _present_options(state: MemberSearchState, candidates: List[MemberCandidate]) -> str:
    options = []
    for index, candidate in enumerate(candidates[:_OPTION_LIMIT], start=1):
        state.candidates_shown.add(candidate.member_random_key)
        options.append(
            f"{index}. {candidate.base_name} — فروشگاه {candidate.shop_id}"
            f" (قیمت حدود {candidate.price:,} تومان)"
        )
    state.pending_question = "candidate_choice"
    message = (
        "چند گزینه نزدیک پیدا کردم:\n" + "\n".join(options) +
        "\nلطفاً شماره گزینه یا شناسه فروشگاه مورد نظرتان را بنویسید."
        " اگر هیچ‌کدام مناسب نیست بگویید \"هیچکدام\" و ویژگی مد نظر را توضیح دهید."
    )
    return message


def _choose_question(state: MemberSearchState, candidates: List[MemberCandidate]) -> str:
    if state.turns_taken == 0 and "broad_intro" not in state.asked_questions:
        state.pending_question = "broad_intro"
        return (
            "برای اینکه سریع‌تر به نتیجه برسیم، لطفاً دسته‌بندی دقیق، حدود بودجه، شهر یا"
            " فروشگاه‌های مد نظر و اینکه ضمانت برایتان مهم است یا خیر را بیان کنید."
        )

    def _field_variation(values: Iterable[object]) -> bool:
        unique = {value for value in values if value is not None}
        return len(unique) > 1

    if (
        "city_id" not in state.excluded_fields
        and "city" not in state.asked_questions
        and _field_variation(candidate.city_id for candidate in candidates)
    ):
        state.pending_question = "city"
        return (
            "آیا شهر یا منطقه خاصی برای خرید مد نظر دارید؟ اگر بله، نام یا شناسه شهر را"
            " بگویید؛ در غیر این صورت بنویسید \"فرقی ندارد\"."
        )

    if (
        "requires_warranty" not in state.excluded_fields
        and "warranty" not in state.asked_questions
        and _field_variation(candidate.has_warranty for candidate in candidates)
    ):
        state.pending_question = "warranty"
        return "آیا حتماً به ضمانت فروشنده نیاز دارید؟ پاسخ بله یا خیر کافی است."

    if (
        "price_range" not in state.excluded_fields
        and "price" not in state.asked_questions
        and _field_variation(candidate.price for candidate in candidates)
    ):
        state.pending_question = "price"
        return "حدود بودجه یا بازه قیمت مد نظرتان برای این محصول چقدر است؟"

    if (
        "brand_id" not in state.excluded_fields
        and "brand" not in state.asked_questions
        and _field_variation(candidate.brand_id for candidate in candidates)
    ):
        state.pending_question = "brand"
        return "آیا برند یا تولیدکننده خاصی مد نظر دارید؟ اگر نه بگویید \"فرقی ندارد\"."

    if (
        "score" not in state.excluded_fields
        and "score" not in state.asked_questions
        and _field_variation(candidate.shop_score for candidate in candidates)
    ):
        state.pending_question = "score"
        return "حداقل امتیاز فروشنده‌ای که مد نظر دارید چقدر است؟"

    state.pending_question = "features"
    return (
        "لطفاً ویژگی‌های شاخص محصول (جنس، ظرفیت، رنگ یا مدل) را بگویید تا بتوانم"
        " گزینه‌ها را دقیق‌تر فیلتر کنم."
    )


@lru_cache(maxsize=1)
def get_multi_turn_manager() -> MultiTurnManager:
    """Return a singleton manager reused across requests."""

    return MultiTurnManager()

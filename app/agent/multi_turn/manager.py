"""Multi-turn conversation manager for member discovery."""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from pydantic import BaseModel

from ..schemas import AgentReply
from .models import ConstraintUpdate, MemberCandidate, MemberFilters, MultiTurnState
from .parser import parse_constraints
from .search import search_members


_MAX_TURNS = 5
_CANDIDATE_DISPLAY_LIMIT = 5
_DIGIT_TRANSLATION = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


@dataclass(frozen=True)
class _Question:
    key: str
    prompt: str
    target_fields: frozenset[str]
    broad: bool = False


_QUESTION_BANK: List[_Question] = [
    _Question(
        key="broad_info",
        prompt=(
            "برای اینکه سریع‌تر عضو دقیق را پیدا کنم لطفاً در یک پیام دسته‌بندی حدودی،"
            " شهر یا شهرهای مدنظرت، محدوده قیمت و ویژگی‌های متمایز (مثل جنس، ظرفیت،"
            " رنگ یا کاربری خاص) را توضیح بده."
        ),
        target_fields=frozenset(
            {
                "category_id",
                "city_id",
                "min_price",
                "max_price",
                "text_queries",
                "requires_warranty",
            }
        ),
        broad=True,
    ),
    _Question(
        key="brand",
        prompt=(
            "برند خاصی مدنظرت هست یا برند اهمیت ندارد؟ اگر برند مشخصی را ترجیح می‌دهی"
            " لطفاً همان را بگو."
        ),
        target_fields=frozenset({"brand_id"}),
    ),
    _Question(
        key="city",
        prompt=(
            "فروشنده در کدام شهر باشد بهتر است؟ اگر شهر اهمیتی ندارد همین را اعلام کن تا"
            " بر اساس ویژگی‌های دیگر جلو برویم."
        ),
        target_fields=frozenset({"city_id"}),
    ),
    _Question(
        key="price",
        prompt="محدوده قیمت مدنظرت برای این محصول چقدر است؟ می‌توانی بازه‌ای به تومان بگویی.",
        target_fields=frozenset({"min_price", "max_price"}),
    ),
    _Question(
        key="warranty",
        prompt="آیا لازم است فروشنده ضمانت ترب داشته باشد یا فرقی نمی‌کند؟",
        target_fields=frozenset({"requires_warranty"}),
    ),
    _Question(
        key="score",
        prompt="حداقل امتیاز فروشنده از ۵ چقدر باشد تا خیال‌ات راحت شود؟",
        target_fields=frozenset({"min_score"}),
    ),
    _Question(
        key="feature",
        prompt=(
            "چه ویژگی یا استفاده‌ی خاصی باید حتماً در محصول باشد؟ هر مشخصه‌ای که به"
            " ذهنت می‌رسد بگو تا در جستجو لحاظ کنم."
        ),
        target_fields=frozenset({"text_queries"}),
    ),
]


class MultiTurnManager:
    """Coordinator that handles scenario 4 conversations."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sessions: dict[str, MultiTurnState] = {}

    async def handle_request(
        self,
        chat_id: str,
        messages: Sequence[BaseModel],
        *,
        session,
    ) -> AgentReply:
        """Process the latest multi-turn request and return the assistant reply."""

        state = await self._get_or_create_state(chat_id)
        new_texts = self._extract_new_messages(state, messages)
        if not new_texts:
            return AgentReply(
                message=state.last_prompt
                or "برای ادامه لطفاً پاسخ خود را به صورت متن ارسال کن.",
            )

        for text in new_texts:
            await self._ingest_user_message(state, text)

        reply = await self._decide_next_action(state, session)

        state.turn_count += 1
        state.last_prompt = reply.message

        if reply.member_random_keys:
            state.completed = True

        await self._finalize_state(chat_id, state)
        return reply

    async def _get_or_create_state(self, chat_id: str) -> MultiTurnState:
        async with self._lock:
            state = self._sessions.get(chat_id)
            if state is None:
                state = MultiTurnState(chat_id=chat_id)
                self._sessions[chat_id] = state
            return state

    async def _finalize_state(self, chat_id: str, state: MultiTurnState) -> None:
        should_clear = state.completed or state.turn_count >= _MAX_TURNS
        async with self._lock:
            if should_clear:
                self._sessions.pop(chat_id, None)
            else:
                self._sessions[chat_id] = state

    def _extract_new_messages(
        self, state: MultiTurnState, messages: Sequence[BaseModel]
    ) -> List[str]:
        new_messages = messages[state.processed_message_count :]
        texts: List[str] = []
        for message in new_messages:
            if getattr(message, "type", None) != "text":
                continue
            content = getattr(message, "content", "")
            stripped = content.strip()
            if stripped:
                texts.append(stripped)
        state.processed_message_count = len(messages)
        return texts

    async def _ingest_user_message(self, state: MultiTurnState, text: str) -> None:
        update = await parse_constraints(state, text)
        self._apply_update(state, update)
        self._handle_candidate_selection(state, text, update)

    def _apply_update(self, state: MultiTurnState, update: ConstraintUpdate) -> None:
        filters = state.filters

        if update.excluded_fields:
            for field in update.excluded_fields:
                self._clear_filter(filters, field)

        if update.cleared_fields:
            for field in update.cleared_fields:
                self._clear_filter(filters, field)

        if update.text_queries:
            filters.add_text_queries(update.text_queries)

        if update.category_id is not None and "category_id" not in filters.excluded_fields:
            filters.category_id = update.category_id
        if update.brand_id is not None and "brand_id" not in filters.excluded_fields:
            filters.brand_id = update.brand_id
        if update.city_id is not None and "city_id" not in filters.excluded_fields:
            filters.city_id = update.city_id

        if update.price_range is not None and "price" not in filters.excluded_fields:
            if update.price_range.min_price is not None:
                filters.min_price = update.price_range.min_price
            if update.price_range.max_price is not None:
                filters.max_price = update.price_range.max_price

        if update.requires_warranty is not None and "requires_warranty" not in filters.excluded_fields:
            filters.requires_warranty = update.requires_warranty

        if update.min_score is not None and "min_score" not in filters.excluded_fields:
            filters.min_score = update.min_score
        if update.max_score is not None and "max_score" not in filters.excluded_fields:
            filters.max_score = update.max_score

        if update.allowed_shop_ids:
            filters.allowed_shop_ids = list(dict.fromkeys(update.allowed_shop_ids))
        if update.preferred_shop_ids:
            filters.preferred_shop_ids = list(dict.fromkeys(update.preferred_shop_ids))

        if update.rejected_candidates:
            state.reset_presented_candidates()
            filters.preferred_shop_ids = []
            filters.allowed_shop_ids = []

        if update.notes:
            filters.other_constraints.setdefault("notes", []).extend(update.notes)

    def _handle_candidate_selection(
        self, state: MultiTurnState, raw_text: str, update: ConstraintUpdate
    ) -> None:
        if not state.presented_candidates:
            state.pending_question_key = None
            return

        selected = None
        if update.selected_member_random_key:
            selected = next(
                (
                    cand
                    for cand in state.presented_candidates
                    if cand.member_random_key == update.selected_member_random_key
                ),
                None,
            )

        if selected is None and update.preferred_shop_ids:
            selected = self._candidate_by_shop(state.presented_candidates, update.preferred_shop_ids)

        if selected is None:
            selected = self._candidate_from_text(state.presented_candidates, raw_text)

        if selected is None and (
            update.rejected_candidates or self._is_rejection_text(raw_text)
        ):
            state.reset_presented_candidates()
            state.pending_question_key = None
            return

        if selected is not None:
            state.filters.other_constraints["selected_member_random_key"] = selected.member_random_key
            state.filters.preferred_shop_ids = [selected.shop_id]
            state.presented_candidates = [selected]
            state.completed = True
        state.pending_question_key = None

    def _candidate_by_shop(
        self, candidates: Sequence[MemberCandidate], shop_ids: Iterable[int]
    ) -> MemberCandidate | None:
        shop_id_set = list(dict.fromkeys(shop_ids))
        for shop_id in shop_id_set:
            match = [c for c in candidates if c.shop_id == shop_id]
            if match:
                return match[0]
        return None

    def _is_rejection_text(self, raw_text: str) -> bool:
        lowered = raw_text.lower()
        return any(token in lowered for token in ["هیچ", "none", "ندارم", "نمی خوام", "نمیخوام"])

    def _candidate_from_text(
        self, candidates: Sequence[MemberCandidate], raw_text: str
    ) -> MemberCandidate | None:
        lowered = raw_text.lower()
        if any(token in lowered for token in ["هیچ", "none", "نیست"]):
            return None

        normalised = raw_text.translate(_DIGIT_TRANSLATION)
        digits = re.findall(r"\d+", normalised)
        for value in digits:
            index = int(value)
            if 1 <= index <= len(candidates):
                return candidates[index - 1]
        for candidate in candidates:
            if candidate.member_random_key and candidate.member_random_key in normalised:
                return candidate
            if str(candidate.shop_id) in normalised:
                return candidate
        return None

    async def _decide_next_action(
        self, state: MultiTurnState, session
    ) -> AgentReply:
        final_round = state.turn_count >= (_MAX_TURNS - 1)

        search_result = await search_members(session, state.filters, limit=20)
        candidates = list(search_result.candidates)
        if candidates:
            state.last_candidates = candidates

        if state.completed and state.presented_candidates:
            choice = state.presented_candidates[0]
            return self._final_reply(choice, reason=None)

        if not candidates:
            if final_round and state.last_candidates:
                choice = state.last_candidates[0]
                state.completed = True
                return self._final_reply(
                    choice,
                    reason="برای اینکه در محدودیت پنج نوبت بمانیم بهترین گزینه قبلی را انتخاب کردم.",
                )
            message = (
                "هیچ گزینه‌ای مطابق با شرایط فعلی پیدا نشد. لطفاً ویژگی شاخص یا محدوده"
                " قیمت دیگری را بگو تا ادامه دهیم."
            )
            state.pending_question_key = None
            return AgentReply(message=message)

        if len(candidates) == 1:
            state.completed = True
            return self._final_reply(candidates[0], reason=None)

        if final_round:
            best_candidate = candidates[0]
            state.completed = True
            return self._final_reply(
                best_candidate,
                reason="برای رعایت محدودیت پنج نوبت دقیق‌ترین گزینه را انتخاب کردم.",
            )

        next_question = self._select_question(state, candidates)
        if next_question is not None:
            state.filters.asked_questions.add(next_question.key)
            state.filters.last_question_key = next_question.key
            state.pending_question_key = next_question.key
            return AgentReply(message=next_question.prompt)

        presentation = self._present_candidates(state, candidates)
        state.filters.last_question_key = "select_candidate"
        state.pending_question_key = "select_candidate"
        return AgentReply(message=presentation)

    def _select_question(
        self, state: MultiTurnState, candidates: Sequence[MemberCandidate]
    ) -> _Question | None:
        filters = state.filters

        if "broad_info" not in filters.asked_questions and "broad_info" not in filters.excluded_fields:
            return _QUESTION_BANK[0]

        brand_values = {c.brand_id for c in candidates if c.brand_id is not None}
        city_values = {c.city_id for c in candidates if c.city_id is not None}
        warranty_values = {c.has_warranty for c in candidates}
        score_values = {math.floor(c.shop_score or 0) for c in candidates if c.shop_score is not None}

        for question in _QUESTION_BANK[1:]:
            if question.key in filters.asked_questions:
                continue
            if question.key in filters.excluded_fields:
                continue
            if question.key == "brand" and filters.brand_id is None and len(brand_values) > 1:
                return question
            if question.key == "city" and filters.city_id is None and len(city_values) > 1:
                return question
            if question.key == "price" and (filters.min_price is None or filters.max_price is None):
                return question
            if question.key == "warranty" and filters.requires_warranty is None and len(warranty_values) > 1:
                return question
            if question.key == "score" and filters.min_score is None and len(score_values) > 1:
                return question
            if question.key == "feature" and len(filters.text_queries) < 4:
                return question
        return None

    def _present_candidates(
        self, state: MultiTurnState, candidates: Sequence[MemberCandidate]
    ) -> str:
        top_candidates = list(candidates[:_CANDIDATE_DISPLAY_LIMIT])
        state.presented_candidates = top_candidates
        state.filters.preferred_shop_ids = []
        state.filters.allowed_shop_ids = []
        shown_keys = {candidate.member_random_key for candidate in top_candidates}
        existing = set(state.filters.candidates_shown)
        state.filters.candidates_shown.extend(sorted(shown_keys - existing))

        lines = [
            "چند گزینه مناسب پیدا کردم. با نوشتن شماره گزینه یا شناسه فروشگاه بگو کدام را می‌خواهی:",
        ]
        for idx, candidate in enumerate(top_candidates, start=1):
            warranty_part = "، با ضمانت ترب" if candidate.has_warranty else ""
            score_part = (
                f"، امتیاز فروشگاه {candidate.shop_score:.1f}"
                if candidate.shop_score is not None
                else ""
            )
            city_part = (
                f"، شهر {candidate.city_name}" if candidate.city_name else ""
            )
            lines.append(
                f"{idx}) {candidate.base_name} – فروشگاه {candidate.shop_id}{city_part}"
                f"، قیمت {candidate.price:,} تومان{warranty_part}{score_part}"
            )
        lines.append("اگر هیچ‌کدام مناسب نیست بگو هیچ‌کدام تا دقیق‌تر جستجو کنم.")
        return "\n".join(lines)

    def _final_reply(self, candidate: MemberCandidate, reason: str | None) -> AgentReply:
        message_parts = []
        if reason:
            message_parts.append(reason)
        message_parts.append(
            f"گزینه پیشنهادی: {candidate.base_name} از فروشگاه {candidate.shop_id}."
        )
        if candidate.city_name:
            message_parts.append(f"شهر: {candidate.city_name}.")
        message_parts.append(f"قیمت: {candidate.price:,} تومان.")
        if candidate.has_warranty:
            message_parts.append("فروشنده ضمانت ترب دارد.")
        if candidate.shop_score is not None:
            message_parts.append(f"امتیاز فروشنده {candidate.shop_score:.1f} از ۵ است.")
        return AgentReply(
            message=" ".join(message_parts),
            member_random_keys=[candidate.member_random_key],
        )

    def _clear_filter(self, filters: MemberFilters, field: str) -> None:
        normalised = field.lower()
        if "category" in normalised:
            filters.category_id = None
            filters.excluded_fields.add("category_id")
        elif "brand" in normalised:
            filters.brand_id = None
            filters.excluded_fields.add("brand_id")
        elif "city" in normalised:
            filters.city_id = None
            filters.excluded_fields.add("city_id")
        elif "price" in normalised:
            filters.min_price = None
            filters.max_price = None
            filters.excluded_fields.add("price")
        elif "warranty" in normalised:
            filters.requires_warranty = None
            filters.excluded_fields.add("requires_warranty")
        elif "score" in normalised:
            filters.min_score = None
            filters.max_score = None
            filters.excluded_fields.add("min_score")


_MANAGER: MultiTurnManager | None = None


def get_multi_turn_manager() -> MultiTurnManager:
    """Return the singleton multi-turn manager instance."""

    global _MANAGER
    if _MANAGER is None:
        _MANAGER = MultiTurnManager()
    return _MANAGER


__all__ = ["MultiTurnManager", "get_multi_turn_manager"]


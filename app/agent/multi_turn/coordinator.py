"""Coordinator orchestrating the scenario 4 multi-agent workflow."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from typing import Dict, Optional

from sqlalchemy import func, select
from pydantic_ai.usage import UsageLimits

from ..dependencies import AgentDependencies
from ..schemas import AgentReply
from .agents import (
    get_candidate_reducer_agent,
    get_clarification_agent,
    get_constraint_extractor_agent,
    get_finaliser_agent,
    get_member_resolver_agent,
    get_search_agent,
)
from ...models import City, Member, Shop
from .schemas import MemberOffer
from .tools import _score_member_offer
from .state import Scenario4ConversationState, _normalise_text as _normalise_city_text


_UNSATISFIED_MESSAGES = {
    "city": "هیچ پیشنهادی مطابق شهرهای {preferred} پیدا نشد؛ نزدیک‌ترین گزینه از شهر دیگری است.",
    "warranty": "پیشنهاد ضمانت‌دار مطابق خواسته پیدا نشد؛ گزینه نهایی بدون ضمانت ارائه می‌شود.",
    "shop_score": "فروشنده‌ای با امتیاز {score} یا بالاتر موجود نبود؛ بهترین گزینه امتیاز پایین‌تری دارد.",
}


def _format_price_note(price_min: int | None, price_max: int | None) -> str | None:
    """Return a human-readable note when price constraints could not be honoured."""

    if price_min is None and price_max is None:
        return None
    if price_min is not None and price_max is not None:
        return (
            "هیچ پیشنهادی در بازه قیمت {low:,} تا {high:,} تومان موجود نبود؛ نزدیک‌ترین"
            " گزینه خارج از این محدوده است."
        ).format(low=price_min, high=price_max)
    if price_min is not None:
        return (
            "هیچ پیشنهادی گران‌تر از {low:,} تومان پیدا نشد؛ نزدیک‌ترین گزینه ارزان‌تر است."
        ).format(low=price_min)
    return (
        "هیچ پیشنهادی ارزان‌تر از {high:,} تومان پیدا نشد؛ نزدیک‌ترین گزینه گران‌تر است."
    ).format(high=price_max)


class _ConversationStore:
    """In-memory registry of ongoing scenario 4 conversations."""

    def __init__(self) -> None:
        self._states: Dict[str, Scenario4ConversationState] = {}
        self._lock = asyncio.Lock()

    async def get(self, chat_id: str) -> Scenario4ConversationState:
        """Return the state for a chat, creating it if required."""

        async with self._lock:
            state = self._states.get(chat_id)
            if state is None:
                state = Scenario4ConversationState(chat_id=chat_id)
                self._states[chat_id] = state
            return state

    async def reset(self, chat_id: str) -> Scenario4ConversationState:
        """Reset the conversation state, typically after completion."""

        async with self._lock:
            state = Scenario4ConversationState(chat_id=chat_id)
            self._states[chat_id] = state
            return state

    async def complete(self, chat_id: str) -> None:
        """Remove the conversation from the active registry."""

        async with self._lock:
            self._states.pop(chat_id, None)


class Scenario4Coordinator:
    """Coordinates the multi-agent workflow for scenario 4."""

    def __init__(self) -> None:
        self._store = _ConversationStore()

    async def handle_turn(
        self,
        *,
        chat_id: str,
        user_message: str,
        deps: AgentDependencies,
        usage_limits: UsageLimits | None = None,
    ) -> AgentReply:
        """Process the latest user message and produce the next assistant reply."""

        state = await self._store.get(chat_id)
        if state.completed:
            state = await self._store.reset(chat_id)

        state.latest_user_message = user_message.strip()

        extraction = await self._run_agent(
            agent_key="constraint_extractor",
            agent_factory=get_constraint_extractor_agent,
            state=state,
            deps=deps,
            usage_limits=usage_limits,
            prompt=self._build_extraction_prompt(state, user_message),
        )
        state.constraints.apply_update(extraction)

        plan = await self._run_agent(
            agent_key="clarification",
            agent_factory=get_clarification_agent,
            state=state,
            deps=deps,
            usage_limits=usage_limits,
            prompt=self._build_clarification_prompt(state),
        )

        action = plan.action
        force_final = state.remaining_turns() <= 1
        if action == "ask_question" and force_final:
            # Forced to decide due to turn budget.
            action = "finalize"

        if action == "ask_question":
            question = plan.question or self._fallback_question(state)
            state.asked_questions.append(question)
            state.next_turn()
            return AgentReply(message=question)

        if action == "search_products":
            await self._search_candidates(state, deps, usage_limits)
            return await self._post_search_response(state, deps, usage_limits, force_final)

        if action == "present_candidates":
            if not state.candidate_products:
                await self._search_candidates(state, deps, usage_limits)
            return await self._present_candidates(state, deps, usage_limits, force_final)

        if action == "resolve_members":
            if not state.locked_base_key and state.candidate_products:
                state.locked_base_key = state.candidate_products[0].base_random_key
            await self._resolve_members(state, deps, usage_limits, force_final)
            return await self._finalise_if_ready(state, deps, usage_limits, force_final)

        # Fall back to finalisation.
        await self._resolve_members(state, deps, usage_limits, force_final)
        return await self._finalise_if_ready(state, deps, usage_limits, force_final)

    async def _post_search_response(
        self,
        state: Scenario4ConversationState,
        deps: AgentDependencies,
        usage_limits: UsageLimits | None,
        force_final: bool,
    ) -> AgentReply:
        """Respond after refreshing candidate products."""

        if not state.candidate_products:
            if force_final:
                await self._resolve_members(state, deps, usage_limits, True)
                return await self._finalise_if_ready(state, deps, usage_limits, True)
            question = self._fallback_question(state)
            state.asked_questions.append(question)
            state.next_turn()
            return AgentReply(message=question)

        if len(state.candidate_products) == 1:
            state.locked_base_key = state.candidate_products[0].base_random_key
            await self._resolve_members(state, deps, usage_limits, force_final)
            return await self._finalise_if_ready(state, deps, usage_limits, force_final)

        if force_final:
            state.locked_base_key = state.candidate_products[0].base_random_key
            await self._resolve_members(state, deps, usage_limits, True)
            return await self._finalise_if_ready(state, deps, usage_limits, True)

        return await self._present_candidates(state, deps, usage_limits, False)

    async def _search_candidates(
        self,
        state: Scenario4ConversationState,
        deps: AgentDependencies,
        usage_limits: UsageLimits | None,
    ) -> None:
        """Invoke the catalogue search agent and update the candidate list."""

        response = await self._run_agent(
            agent_key="search",
            agent_factory=get_search_agent,
            state=state,
            deps=deps,
            usage_limits=usage_limits,
            prompt=self._build_search_prompt(state),
        )
        state.candidate_products = list(response.candidates)
        if state.candidate_products:
            state.locked_base_key = state.candidate_products[0].base_random_key

    async def _present_candidates(
        self,
        state: Scenario4ConversationState,
        deps: AgentDependencies,
        usage_limits: UsageLimits | None,
        force_final: bool,
    ) -> AgentReply:
        """Ask the user to pick among multiple candidate base products."""

        if not state.candidate_products:
            if force_final:
                await self._resolve_members(state, deps, usage_limits, True)
                return await self._finalise_if_ready(state, deps, usage_limits, True)
            question = self._fallback_question(state)
            state.asked_questions.append(question)
            state.next_turn()
            return AgentReply(message=question)

        if force_final:
            state.locked_base_key = state.candidate_products[0].base_random_key
            await self._resolve_members(state, deps, usage_limits, True)
            return await self._finalise_if_ready(state, deps, usage_limits, True)

        top_candidates = state.candidate_products[:3]
        reducer = await self._run_agent(
            agent_key="candidate_reducer",
            agent_factory=get_candidate_reducer_agent,
            state=state,
            deps=deps,
            usage_limits=usage_limits,
            prompt=self._build_candidate_prompt(state, top_candidates),
        )
        state.asked_questions.append(reducer.message)
        state.next_turn()
        return AgentReply(message=reducer.message)

    async def _resolve_members(
        self,
        state: Scenario4ConversationState,
        deps: AgentDependencies,
        usage_limits: UsageLimits | None,
        force: bool,
    ) -> None:
        """Fetch member offers when a base product has been chosen."""

        if not state.locked_base_key:
            return
        response = await self._run_agent(
            agent_key="member_resolver",
            agent_factory=get_member_resolver_agent,
            state=state,
            deps=deps,
            usage_limits=usage_limits,
            prompt=self._build_member_prompt(state, force_finalization=force),
        )
        state.candidate_offers = list(response.offers)
        if state.candidate_offers:
            state.finalized_member_key = state.candidate_offers[0].member_random_key
            return

        if force:
            fallback_offer, notes = await self._fallback_member_offer(state, deps)
            if fallback_offer:
                state.candidate_offers = [fallback_offer]
                state.finalized_member_key = fallback_offer.member_random_key
                for note in notes:
                    state.record_unsatisfied(note)

    async def _finalise_if_ready(
        self,
        state: Scenario4ConversationState,
        deps: AgentDependencies,
        usage_limits: UsageLimits | None,
        force: bool,
    ) -> AgentReply:
        """Produce the final answer once a member has been selected."""

        if not state.finalized_member_key:
            # Attempt to pick a member even without explicit offers.
            if state.candidate_offers:
                state.finalized_member_key = state.candidate_offers[0].member_random_key
            if not state.finalized_member_key and force:
                fallback_offer, notes = await self._fallback_member_offer(state, deps)
                if fallback_offer:
                    state.candidate_offers = [fallback_offer]
                    state.finalized_member_key = fallback_offer.member_random_key
                    for note in notes:
                        state.record_unsatisfied(note)

        if not state.finalized_member_key:
            if force:
                failure_message = self._build_failure_message(state)
                state.completed = True
                await self._store.complete(state.chat_id)
                state.next_turn()
                return AgentReply(message=failure_message)
            # As a last resort, ask a broad clarifying question when budget remains.
            question = self._fallback_question(state)
            state.asked_questions.append(question)
            state.next_turn()
            return AgentReply(message=question)

        final_message = await self._run_agent(
            agent_key="finaliser",
            agent_factory=get_finaliser_agent,
            state=state,
            deps=deps,
            usage_limits=usage_limits,
            prompt=self._build_final_prompt(state),
        )
        final_key = state.finalized_member_key
        state.completed = True
        await self._store.complete(state.chat_id)
        state.next_turn()
        return AgentReply(
            message=final_message.message,
            member_random_keys=[final_key] if final_key else [],
        )

    async def _fallback_member_offer(
        self,
        state: Scenario4ConversationState,
        deps: AgentDependencies,
    ) -> tuple[MemberOffer | None, list[str]]:
        """Best-effort lookup when the resolver could not find a compliant offer."""

        constraints = state.constraints
        dismissed = set(constraints.dismissed_aspects)
        price_min = constraints.price_min if not constraints.aspect_dismissed("price") else None
        price_max = constraints.price_max if not constraints.aspect_dismissed("price") else None
        city_preferences = (
            sorted(constraints.city_preferences)
            if constraints.city_preferences and not constraints.aspect_dismissed("city")
            else []
        )
        require_warranty = (
            True
            if constraints.require_warranty is True
            and not constraints.aspect_dismissed("warranty")
            else False
        )
        scoring_require_warranty = (
            True
            if constraints.require_warranty is True
            and not constraints.aspect_dismissed("warranty")
            else None
        )
        min_shop_score = (
            constraints.min_shop_score
            if constraints.min_shop_score is not None
            and not constraints.aspect_dismissed("shop_score")
            else None
        )

        price_filters_present = price_min is not None or price_max is not None
        price_options = [True, False] if price_filters_present else [False]
        city_options: list[str | None] = city_preferences + [None] if city_preferences else [None]
        warranty_options = [True, False] if require_warranty else [False]
        score_options = [True, False] if min_shop_score is not None else [False]

        candidate_keys: list[str] = []
        if state.locked_base_key:
            candidate_keys.append(state.locked_base_key)
        for product in state.candidate_products:
            if product.base_random_key and product.base_random_key not in candidate_keys:
                candidate_keys.append(product.base_random_key)

        if not candidate_keys:
            return None, []

        async with deps.session_factory() as session:
            for candidate_key in candidate_keys:
                base_stmt = (
                    select(
                        Member.random_key,
                        Member.price,
                        Shop.id,
                        Shop.has_warranty,
                        Shop.score,
                        City.name,
                    )
                    .join(Shop, Shop.id == Member.shop_id)
                    .join(City, City.id == Shop.city_id, isouter=True)
                    .where(Member.base_random_key == candidate_key)
                )

                for use_price in price_options:
                    for city_value in city_options:
                        for use_warranty in warranty_options:
                            for use_score in score_options:
                                stmt = base_stmt
                                notes: list[str] = []

                                if candidate_key != state.locked_base_key and state.locked_base_key:
                                    notes.append(
                                        "محصول اول پیشنهادی موجود نبود؛ نزدیک‌ترین گزینه از محصول دیگری انتخاب شد."
                                    )

                                if use_price and price_min is not None:
                                    stmt = stmt.where(Member.price >= price_min)
                                if use_price and price_max is not None:
                                    stmt = stmt.where(Member.price <= price_max)
                                if (
                                    not use_price
                                    and price_filters_present
                                    and not constraints.aspect_dismissed("price")
                                ):
                                    price_note = _format_price_note(price_min, price_max)
                                    if price_note:
                                        notes.append(price_note)

                                if city_value is not None:
                                    stmt = stmt.where(
                                        func.lower(City.name) == _normalise_city_text(city_value)
                                    )
                                elif city_preferences and not constraints.aspect_dismissed("city"):
                                    notes.append(
                                        _UNSATISFIED_MESSAGES["city"].format(
                                            preferred="، ".join(city_preferences)
                                        )
                                    )

                                if use_warranty and require_warranty:
                                    stmt = stmt.where(Shop.has_warranty.is_(True))
                                elif require_warranty and not constraints.aspect_dismissed("warranty"):
                                    notes.append(_UNSATISFIED_MESSAGES["warranty"])

                                if use_score and min_shop_score is not None:
                                    stmt = stmt.where(Shop.score >= float(min_shop_score))
                                elif (
                                    min_shop_score is not None
                                    and not constraints.aspect_dismissed("shop_score")
                                ):
                                    notes.append(
                                        _UNSATISFIED_MESSAGES["shop_score"].format(
                                            score=f"{min_shop_score:g}"
                                        )
                                    )

                                stmt = stmt.order_by(Member.price.asc(), Shop.score.desc()).limit(1)
                                result = await session.execute(stmt)
                                row = result.first()
                                if not row:
                                    continue

                                member_random_key, price, shop_id, has_warranty, score, city_name = row
                                matched_constraints, match_score = _score_member_offer(
                                    price=int(price),
                                    has_warranty=bool(has_warranty),
                                    shop_score=score,
                                    city_name=city_name,
                                    price_min=price_min,
                                    price_max=price_max,
                                    require_warranty=scoring_require_warranty,
                                    min_shop_score=min_shop_score,
                                    city=city_value,
                                    dismissed=dismissed,
                                )
                                offer = MemberOffer(
                                    member_random_key=member_random_key,
                                    shop_id=int(shop_id),
                                    price=int(price),
                                    has_warranty=bool(has_warranty),
                                    shop_score=Decimal(str(score)) if score is not None else None,
                                    city_name=city_name,
                                    matched_constraints=matched_constraints,
                                    match_score=match_score,
                                )
                                state.locked_base_key = candidate_key
                                return offer, notes

            # Absolute fallback: pick the cheapest available offer across the catalogue.
            fallback_stmt = (
                select(
                    Member.random_key,
                    Member.base_random_key,
                    Member.price,
                    Shop.id,
                    Shop.has_warranty,
                    Shop.score,
                    City.name,
                )
                .join(Shop, Shop.id == Member.shop_id)
                .join(City, City.id == Shop.city_id, isouter=True)
            )
            if price_min is not None:
                fallback_stmt = fallback_stmt.where(Member.price >= price_min)
            if price_max is not None:
                fallback_stmt = fallback_stmt.where(Member.price <= price_max)
            fallback_stmt = fallback_stmt.order_by(Member.price.asc(), Shop.score.desc()).limit(1)
            result = await session.execute(fallback_stmt)
            row = result.first()
            if row:
                (
                    member_random_key,
                    base_random_key,
                    price,
                    shop_id,
                    has_warranty,
                    score,
                    city_name,
                ) = row
                matched_constraints, match_score = _score_member_offer(
                    price=int(price),
                    has_warranty=bool(has_warranty),
                    shop_score=score,
                    city_name=city_name,
                    price_min=price_min,
                    price_max=price_max,
                    require_warranty=scoring_require_warranty,
                    min_shop_score=min_shop_score,
                    city=city_preferences[0] if city_preferences else None,
                    dismissed=dismissed,
                )
                offer = MemberOffer(
                    member_random_key=member_random_key,
                    shop_id=int(shop_id),
                    price=int(price),
                    has_warranty=bool(has_warranty),
                    shop_score=Decimal(str(score)) if score is not None else None,
                    city_name=city_name,
                    matched_constraints=matched_constraints,
                    match_score=match_score,
                )
                notes = [
                    "برای پاسخ‌گویی مجبور شدم نزدیک‌ترین فروشنده موجود در کل کاتالوگ را معرفی کنم."
                ]
                if (
                    price_filters_present
                    and not constraints.aspect_dismissed("price")
                    and (
                        (price_min is not None and price < price_min)
                        or (price_max is not None and price > price_max)
                    )
                ):
                    price_note = _format_price_note(price_min, price_max)
                    if price_note:
                        notes.append(price_note)
                state.locked_base_key = base_random_key
                return offer, notes

        return None, []

    async def _run_agent(
        self,
        *,
        agent_key: str,
        agent_factory,
        state: Scenario4ConversationState,
        deps: AgentDependencies,
        prompt: str,
        usage_limits: UsageLimits | None,
    ):
        """Execute an agent while preserving its conversation history."""

        agent = agent_factory()
        existing_history = state.agent_histories.get(agent_key)
        message_history = list(existing_history) if existing_history else None
        result = await agent.run(
            user_prompt=prompt,
            deps=deps,
            message_history=message_history,
            usage_limits=usage_limits,
        )
        new_messages = list(result.new_messages())
        if existing_history:
            updated_history = list(existing_history)
            if new_messages:
                updated_history.extend(new_messages)
        else:
            updated_history = new_messages or list(result.all_messages())
        state.agent_histories[agent_key] = updated_history
        return result.output

    def _build_extraction_prompt(
        self, state: Scenario4ConversationState, latest_user_message: str
    ) -> str:
        context = {
            "previous_summaries": state.constraints.summaries[-3:],
            "questions_asked": state.asked_questions[-3:],
        }
        return (
            "Extract structured constraints from the latest user reply."
            "\nContext: "
            f"{json.dumps(context, ensure_ascii=False)}\n"
            f"Latest user message: {latest_user_message}"
        )

    def _build_clarification_prompt(self, state: Scenario4ConversationState) -> str:
        snapshot = state.constraints.snapshot()
        payload = {
            "constraints": snapshot,
            "candidates_found": len(state.candidate_products),
            "locked_base_key": state.locked_base_key,
            "asked_questions": state.asked_questions[-3:],
            "remaining_turns": state.remaining_turns(),
        }
        return (
            "Decide the next action using the structured snapshot below."
            "\n" + json.dumps(payload, ensure_ascii=False)
        )

    def _build_search_prompt(self, state: Scenario4ConversationState) -> str:
        payload = {
            "constraints": state.constraints.snapshot(),
            "previous_candidates": [
                candidate.model_dump(mode="json") for candidate in state.candidate_products[:3]
            ],
        }
        return (
            "Use these constraints to retrieve candidate base products via the filter tool."
            "\n" + json.dumps(payload, ensure_ascii=False)
        )

    def _build_candidate_prompt(
        self,
        state: Scenario4ConversationState,
        candidates,
    ) -> str:
        payload = [candidate.model_dump(mode="json") for candidate in candidates]
        return (
            "Compare the following candidate base products and ask the user to choose."
            "\n" + json.dumps(payload, ensure_ascii=False)
        )

    def _build_member_prompt(
        self, state: Scenario4ConversationState, *, force_finalization: bool
    ) -> str:
        payload = {
            "base_random_key": state.locked_base_key,
            "constraints": state.constraints.snapshot(),
            "dismissed_aspects": sorted(state.constraints.dismissed_aspects),
            "force_finalization": force_finalization,
            "unsatisfied_requirements": list(state.unsatisfied_requirements),
        }
        return (
            "Retrieve member offers for the resolved base product using the filter tool."
            "\n" + json.dumps(payload, ensure_ascii=False)
        )

    def _build_final_prompt(self, state: Scenario4ConversationState) -> str:
        offer: Optional[MemberOffer] = None
        for candidate in state.candidate_offers:
            if candidate.member_random_key == state.finalized_member_key:
                offer = candidate
                break
        payload = {
            "member_random_key": state.finalized_member_key,
            "base_random_key": state.locked_base_key,
            "offer": offer.model_dump(mode="json") if offer else None,
            "constraints": state.constraints.snapshot(),
            "unsatisfied_requirements": list(state.unsatisfied_requirements),
        }
        return (
            "Summarise the final recommendation and restate the selected member key once."
            "\n" + json.dumps(payload, ensure_ascii=False)
        )

    def _build_failure_message(self, state: Scenario4ConversationState) -> str:
        """Craft a polite apology when no member can be resolved."""

        base = (
            "متأسفانه با وجود جست‌وجوی کامل نتوانستم فروشنده‌ای پیدا کنم که دقیقاً با شرایط شما منطبق باشد."
        )
        if state.unsatisfied_requirements:
            details = "؛ ".join(state.unsatisfied_requirements)
            base += f" مواردی که رعایت نشد: {details}."
        base += " لطفاً اگر مایل هستید، شرایط را کمی تغییر دهید یا اطلاعات بیشتری بدهید تا بتوانم دوباره تلاش کنم."
        return base

    def _fallback_question(self, state: Scenario4ConversationState) -> str:
        """Return a conservative follow-up question when no better option exists."""

        constraints = state.constraints
        if not constraints.category_hint:
            return "برای چه نوع محصول یا دسته‌ای به دنبال گزینه هستید و چه بودجه یا ویژگی شاخصی برایتان مهم است؟"
        if not constraints.brand_preferences and not constraints.aspect_dismissed("brand"):
            return "برند یا سبک خاصی مدنظرتان است تا دقیق‌تر جستجو کنم؟"
        if (
            (constraints.price_min is None or constraints.price_max is None)
            and not constraints.aspect_dismissed("price")
        ):
            return "حدود بودجه مورد نظرتان برای این خرید چقدر است؟"
        if constraints.require_warranty is None and not constraints.aspect_dismissed("warranty"):
            return "گارانتی برایتان اهمیت دارد یا می‌توانم بدون گارانتی هم جستجو کنم؟"
        if not constraints.city_preferences and not constraints.aspect_dismissed("city"):
            return "ترجیح می‌دهید محصول از کدام شهر ارسال شود؟"
        if (
            not constraints.required_features.values()
            and not constraints.optional_features.values()
            and not constraints.aspect_dismissed("features")
        ):
            return "کدام ویژگی یا جنس برایتان اهمیت دارد تا گزینه‌ها را دقیق‌تر پیدا کنم؟"
        return "اگر نکته دیگری مهم است بفرمایید تا سریع‌تر جمع‌بندی کنم."


_COORDINATOR: Scenario4Coordinator | None = None


def get_scenario4_coordinator() -> Scenario4Coordinator:
    """Return a singleton coordinator instance for scenario 4."""

    global _COORDINATOR
    if _COORDINATOR is None:
        _COORDINATOR = Scenario4Coordinator()
    return _COORDINATOR


__all__ = ["Scenario4Coordinator", "get_scenario4_coordinator"]


"""Coordinator orchestrating the scenario 4 multi-agent workflow."""

from __future__ import annotations

import asyncio
import json
from typing import Dict, Optional
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
from .schemas import MemberOffer
from .state import Scenario4ConversationState


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
        if action == "ask_question" and state.remaining_turns() <= 1:
            # Forced to decide due to turn budget.
            action = "finalize"

        if action == "ask_question":
            question = plan.question or self._fallback_question(state)
            state.asked_questions.append(question)
            state.next_turn()
            return AgentReply(message=question)

        if action == "search_products":
            await self._search_candidates(state, deps, usage_limits)
            return await self._post_search_response(state, deps, usage_limits)

        if action == "present_candidates":
            if not state.candidate_products:
                await self._search_candidates(state, deps, usage_limits)
            return await self._present_candidates(state, deps, usage_limits)

        if action == "resolve_members":
            if not state.locked_base_key and state.candidate_products:
                state.locked_base_key = state.candidate_products[0].base_random_key
            await self._resolve_members(state, deps, usage_limits)
            return await self._finalise_if_ready(state, deps, usage_limits)

        # Fall back to finalisation.
        await self._resolve_members(state, deps, usage_limits)
        return await self._finalise_if_ready(state, deps, usage_limits)

    async def _post_search_response(
        self,
        state: Scenario4ConversationState,
        deps: AgentDependencies,
        usage_limits: UsageLimits | None,
    ) -> AgentReply:
        """Respond after refreshing candidate products."""

        if not state.candidate_products:
            question = self._fallback_question(state)
            state.asked_questions.append(question)
            state.next_turn()
            return AgentReply(message=question)

        if len(state.candidate_products) == 1:
            state.locked_base_key = state.candidate_products[0].base_random_key
            await self._resolve_members(state, deps, usage_limits)
            return await self._finalise_if_ready(state, deps, usage_limits)

        return await self._present_candidates(state, deps, usage_limits)

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
    ) -> AgentReply:
        """Ask the user to pick among multiple candidate base products."""

        if not state.candidate_products:
            question = self._fallback_question(state)
            state.asked_questions.append(question)
            state.next_turn()
            return AgentReply(message=question)

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
            prompt=self._build_member_prompt(state),
        )
        state.candidate_offers = list(response.offers)
        if state.candidate_offers:
            state.finalized_member_key = state.candidate_offers[0].member_random_key

    async def _finalise_if_ready(
        self,
        state: Scenario4ConversationState,
        deps: AgentDependencies,
        usage_limits: UsageLimits | None,
    ) -> AgentReply:
        """Produce the final answer once a member has been selected."""

        if not state.finalized_member_key:
            # Attempt to pick a member even without explicit offers.
            if state.candidate_offers:
                state.finalized_member_key = state.candidate_offers[0].member_random_key
            if not state.finalized_member_key and state.candidate_offers:
                state.finalized_member_key = state.candidate_offers[0].member_random_key

        if not state.finalized_member_key:
            # As a last resort, ask a broad clarifying question.
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
        state.completed = True
        await self._store.complete(state.chat_id)
        state.next_turn()
        return AgentReply(
            message=final_message.message,
            member_random_keys=[final_message.member_random_key],
        )

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
        history = state.agent_histories.setdefault(agent_key, [])
        result = await agent.run(
            user_prompt=prompt,
            deps=deps,
            message_history=list(history) if history else None,
            usage_limits=usage_limits,
        )
        new_messages = result.new_messages()
        if new_messages:
            history.extend(new_messages)
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

    def _build_member_prompt(self, state: Scenario4ConversationState) -> str:
        payload = {
            "base_random_key": state.locked_base_key,
            "constraints": state.constraints.snapshot(),
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
        }
        return (
            "Summarise the final recommendation and restate the selected member key once."
            "\n" + json.dumps(payload, ensure_ascii=False)
        )

    def _fallback_question(self, state: Scenario4ConversationState) -> str:
        """Return a conservative follow-up question when no better option exists."""

        if not state.constraints.category_hint:
            return "برای چه نوع محصول یا دسته‌ای به دنبال گزینه هستید؟"
        if not state.constraints.brand_preferences:
            return "برند یا سبک خاصی مدنظرتان است تا دقیق‌تر جستجو کنم؟"
        if state.constraints.price_min is None or state.constraints.price_max is None:
            return "حدود بودجه مورد نظرتان برای این خرید چقدر است؟"
        return "کدام ویژگی برایتان مهم‌تر است تا دقیق‌تر جستجو کنم؟"


_COORDINATOR: Scenario4Coordinator | None = None


def get_scenario4_coordinator() -> Scenario4Coordinator:
    """Return a singleton coordinator instance for scenario 4."""

    global _COORDINATOR
    if _COORDINATOR is None:
        _COORDINATOR = Scenario4Coordinator()
    return _COORDINATOR


__all__ = ["Scenario4Coordinator", "get_scenario4_coordinator"]


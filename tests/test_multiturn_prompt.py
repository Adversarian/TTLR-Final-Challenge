"""Assertions about the multi-turn system prompt contract."""

from __future__ import annotations

from app.agent.multiturn.prompts import MULTI_TURN_PROMPT


def test_multi_turn_prompt_requests_clarification_on_empty_results() -> None:
    """The prompt should avoid retry loops and ask for clarification on empty results."""

    prompt_lower = MULTI_TURN_PROMPT.lower()
    assert "relaxation" not in prompt_lower
    assert "count = 0" in MULTI_TURN_PROMPT
    assert "clarifying" in prompt_lower
    assert "at most once per turn" in prompt_lower

"""Assertions about the multi-turn system prompt contract."""

from __future__ import annotations

from app.agent.multiturn.prompts import MULTI_TURN_PROMPT


def test_multi_turn_prompt_outlines_turn_structure() -> None:
    """The prompt should describe the clarification flow and option presentation timing."""

    prompt_lower = MULTI_TURN_PROMPT.lower()
    assert "count = 0" in MULTI_TURN_PROMPT
    assert "clarifying" in prompt_lower
    assert "at most once per turn" in prompt_lower
    assert "turn four" in prompt_lower
    assert "present up to five" in prompt_lower

"""Generate deterministic system prompts for the assistant agent."""
from __future__ import annotations

from textwrap import dedent

from pydantic_ai.tools import RunContext

from .context import AgentDependencies


def build_system_prompt(ctx: RunContext[AgentDependencies]) -> str:
    """Return the system instructions tailored to the current chat state."""

    state = ctx.deps.state
    memory_section = ""
    if state.last_base_random_key:
        memory_section = (
            f"Memory: last_base_random_key={state.last_base_random_key}"
        )
        if state.last_query:
            memory_section += f", last_query={state.last_query}"
    prompt = dedent(
        f"""
        You are a concise shopping assistant operating under strict latency constraints.
        Follow these principles:
        - Deterministic output (temperature 0) with short sentences.
        - Exactly one tool call per user turn; prefer using memory when the base key is already known.
        - If input lacks details required to solve the task, ask one short clarification question instead of calling any tool.
        - When resolving a product, respond with message=null and a single base_random_key.
        - When providing attribute values, respond with the normalized value text in message and no random keys.
        - When returning numeric seller statistics, the message must contain digits only (no words or commas).
        - Never invent random keys or facts.
        - Use trigram/FTS ProductResolve for base identification, FeatureLookup for attributes, and SellerStats for numeric shop metrics.
        - Honour protocol sanity commands exactly: "ping", "return base random key:", "return member random key:".
        - Keep responses under 60 words.
        {memory_section}
        """
    ).strip()
    return prompt

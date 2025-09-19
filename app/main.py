"""FastAPI application wiring together configuration, agent, and storage."""
from __future__ import annotations

import logging
from typing import List

from fastapi import FastAPI, HTTPException

from .agent.agent import get_agent
from .agent.context import AgentDependencies
from .agent.models import AgentResponse
from .api.models import ChatMessage, ChatRequest, ChatResponse
from .config import get_settings
from .db import db_pool
from .logging import ConversationLogger
from .memory import ChatMemory, ChatMessageRecord
from .protocol import apply_protocol_guards

logger = logging.getLogger(__name__)
app = FastAPI(title="TTLR Shopping Assistant")
settings = get_settings()
conversation_logger = ConversationLogger(settings.conversation_log_path)
chat_memory = ChatMemory()


@app.on_event("startup")
async def on_startup() -> None:
    """Initialise connections and shared resources."""

    await db_pool.connect(settings.database_url)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    """Release pooled connections."""

    await db_pool.close()


def _format_history(records: List[ChatMessageRecord]) -> List[str]:
    lines: List[str] = []
    for record in records[-12:]:
        role = "User" if record.role == "user" else "Assistant"
        prefix = f"{role} ({record.message_type})" if record.message_type != "text" else role
        lines.append(f"{prefix}: {record.content}")
    return lines


def _format_new_messages(messages: List[ChatMessage]) -> List[str]:
    formatted: List[str] = []
    for message in messages:
        if message.type == "text":
            formatted.append(f"User: {message.content}")
        else:
            formatted.append(f"User (image): {message.content}")
    return formatted


def _prepare_prompt(previous: List[ChatMessageRecord], current: List[ChatMessage]) -> str:
    history_lines = _format_history(previous)
    new_lines = _format_new_messages(current)
    segments: List[str] = []
    if history_lines:
        segments.append("Conversation to date:")
        segments.extend(history_lines)
    segments.append("Latest user input:")
    segments.extend(new_lines)
    return "\n".join(segments)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """Main entry-point handling conversation turns."""

    chat_id = payload.chat_id
    incoming_messages = payload.messages
    if not incoming_messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    state = chat_memory.get(chat_id)
    previous_history = list(state.history)

    for message in incoming_messages:
        rendered = message.content if message.type == "text" else f"[image] {message.content}"
        chat_memory.append_history(chat_id, "user", rendered, message_type=message.type)
        await conversation_logger.log(
            {
                "chat_id": chat_id,
                "direction": "incoming",
                "message_type": message.type,
                "content": message.content,
            }
        )

    latest_message = incoming_messages[-1]
    if latest_message.type == "text":
        guard_response = apply_protocol_guards(latest_message.content)
        if guard_response is not None:
            if guard_response.message:
                chat_memory.append_history(chat_id, "assistant", guard_response.message, message_type="text")
            await conversation_logger.log(
                {
                    "chat_id": chat_id,
                    "direction": "system",
                    "response": guard_response.model_dump(),
                }
            )
            return guard_response

    user_prompt = _prepare_prompt(previous_history, incoming_messages)

    agent = get_agent()
    deps = AgentDependencies(chat_id=chat_id, database=db_pool, state=state)

    try:
        result = await agent.run(user_prompt=user_prompt, deps=deps)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.exception("Agent execution failed")
        raise HTTPException(status_code=500, detail="agent failure") from exc

    agent_output: AgentResponse = result.output
    response = ChatResponse(
        message=agent_output.message,
        base_random_keys=agent_output.base_random_keys,
        member_random_keys=agent_output.member_random_keys,
    )

    if response.base_random_keys:
        chat_memory.update_base(chat_id, response.base_random_keys[0], latest_message.content)
    elif response.message:
        chat_memory.update_base(chat_id, state.last_base_random_key, latest_message.content)

    if response.message is not None:
        chat_memory.append_history(chat_id, "assistant", response.message)
    await conversation_logger.log(
        {
            "chat_id": chat_id,
            "direction": "outgoing",
            "response": response.model_dump(),
        }
    )

    return response

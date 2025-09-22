"""Main FastAPI application for the shopping assistant."""

from decimal import Decimal, InvalidOperation
from typing import List, Literal, Optional

from pydantic_ai.usage import UsageLimits
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from sqlalchemy.ext.asyncio import AsyncSession

from .agent import AgentDependencies, get_agent
from .db import AsyncSessionLocal, get_session


class ChatMessage(BaseModel):
    """Represents a single message exchanged in a chat session."""

    type: Literal["text", "image"]
    content: str


class ChatRequest(BaseModel):
    """Payload sent to the `/chat` endpoint."""

    chat_id: str
    messages: List[ChatMessage]


class ChatResponse(BaseModel):
    """Response schema returned by the `/chat` endpoint."""

    message: Optional[str] = None
    base_random_keys: Optional[List[str]] = None
    member_random_keys: Optional[List[str]] = None


app = FastAPI(title="Shopping Assistant API", version="0.1.0")


def _extract_key(command_prefix: str, message: str) -> Optional[str]:
    """Extract a random key following the provided command prefix."""

    lower_prefix = command_prefix.lower()
    lower_message = message.lower()
    if not lower_message.startswith(lower_prefix):
        return None
    parts = message.split(":", maxsplit=1)
    if len(parts) != 2:
        return None
    return parts[1].strip()


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest, session: AsyncSession = Depends(get_session)
) -> ChatResponse:
    """Handle chat interactions with the assistant.

    For scenario 0 the handler returns deterministic responses that allow the
    judge to verify that the API is reachable and well-formed.
    """

    if not request.messages:
        return ChatResponse(message="No messages provided.")

    latest_message = request.messages[-1]
    latest_content = latest_message.content.strip()
    latest_content_lower = latest_content.lower()

    if latest_content_lower == "ping" or request.chat_id == "sanity-check-ping":
        return ChatResponse(message="pong")

    base_key = _extract_key("return base random key:", latest_content)
    if base_key:
        return ChatResponse(base_random_keys=[base_key])

    member_key = _extract_key("return member random key:", latest_content)
    if member_key:
        return ChatResponse(member_random_keys=[member_key])

    if latest_message.type == "image":
        return ChatResponse(
            message="Image messages are not supported yet. Please send text instructions."
        )

    text_messages = [
        message.content.strip()
        for message in request.messages
        if message.type == "text" and message.content.strip()
    ]
    if not text_messages:
        return ChatResponse(message="No textual message found in the request.")

    aggregated_prompt = "\n\n".join(text_messages)

    agent = get_agent()
    deps = AgentDependencies(session=session, session_factory=AsyncSessionLocal)

    try:
        result = await agent.run(
            user_prompt=aggregated_prompt,
            deps=deps,
            usage_limits=UsageLimits(
                request_limit=20,
                tool_calls_limit=10,
            ),
        )
    except Exception as exc:  # pragma: no cover - defensive logging path
        raise HTTPException(status_code=500, detail="Agent execution failed.") from exc

    reply = result.output.clipped()

    message = reply.message
    if reply.numeric_answer is not None:
        try:
            numeric_value = Decimal(reply.numeric_answer)
        except (InvalidOperation, TypeError) as exc:  # pragma: no cover - sanity guard
            raise HTTPException(
                status_code=500, detail="Agent returned a non-numeric statistic."
            ) from exc
        if not numeric_value.is_finite():
            raise HTTPException(
                status_code=500, detail="Agent returned a non-finite statistic."
            )
        message = format(numeric_value.normalize(), "f")

    return ChatResponse(
        message=message,
        base_random_keys=reply.base_random_keys or None,
        member_random_keys=reply.member_random_keys or None,
    )


__all__ = [
    "app",
    "chat_endpoint",
]

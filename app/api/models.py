"""Pydantic schemas for HTTP API payloads."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Represents an incoming user message."""

    type: Literal["text", "image"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    """Payload accepted by the /chat endpoint."""

    chat_id: str = Field(..., min_length=1)
    messages: list[ChatMessage] = Field(..., min_length=1)


class ChatResponse(BaseModel):
    """Response contract for the /chat endpoint."""

    message: Optional[str] = None
    base_random_keys: Optional[list[str]] = None
    member_random_keys: Optional[list[str]] = None

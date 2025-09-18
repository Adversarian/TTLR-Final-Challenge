from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    type: Literal["text"]
    content: str = Field(..., description="Body of the user or assistant message.")


class ChatRequest(BaseModel):
    chat_id: str
    messages: List[ChatMessage]


class ChatResponse(BaseModel):
    message: Optional[str] = None
    base_random_keys: Optional[List[str]] = None
    member_random_keys: Optional[List[str]] = None

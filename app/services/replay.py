from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List
from uuid import uuid4

from app.models.chat import ChatRequest, ChatResponse


_SAFE_ID_PATTERN = re.compile(r"[^a-zA-Z0-9._-]")


def _safe_chat_id(chat_id: str) -> str:
    sanitized = _SAFE_ID_PATTERN.sub("_", chat_id.strip()) if chat_id else ""
    return sanitized or f"chat_{uuid4().hex}"


def _entry_path(log_dir: str, chat_id: str) -> Path:
    base = Path(log_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{_safe_chat_id(chat_id)}.jsonl"


def log_interaction(
    request: ChatRequest, response: ChatResponse, log_dir: str | None
) -> None:
    if not log_dir:
        return

    path = _entry_path(log_dir, request.chat_id)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chat_id": request.chat_id,
        "request": request.model_dump(mode="json"),
        "response": response.model_dump(mode="json"),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_conversation(chat_id: str, log_dir: str) -> List[dict]:
    path = _entry_path(log_dir, chat_id)
    if not path.exists():
        return []
    entries: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def iter_requests(chat_id: str, log_dir: str) -> Iterator[ChatRequest]:
    for entry in load_conversation(chat_id, log_dir):
        payload = entry.get("request")
        if payload:
            yield ChatRequest.model_validate(payload)


__all__ = ["log_interaction", "load_conversation", "iter_requests"]

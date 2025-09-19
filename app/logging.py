"""Conversation logging utilities."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class ConversationLogger:
    """Append structured chat events into a JSON lines file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.touch(exist_ok=True)

    async def log(self, payload: Dict[str, Any]) -> None:
        """Persist a log payload with automatic timestamping."""

        record = {
            **payload,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        json_line = json.dumps(record, ensure_ascii=False)
        async with self._lock:
            await asyncio.to_thread(self._append_line, json_line)

    def _append_line(self, line: str) -> None:
        """Write a line to the log file (runs inside a thread)."""

        with self._path.open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")

"""Structured logging helpers for judge-triggered chat requests."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from pydantic import BaseModel

from ..config import settings

logger = logging.getLogger(__name__)

_LOGGED_CHAT_PREFIX = "t"


class _LogSession:
    """Collects incoming requests until the session is persisted to disk."""

    def __init__(self, directory: Path, started_at: datetime) -> None:
        self._directory = directory
        self.started_at = started_at
        self.last_activity = started_at
        self.file_path = directory / f"judge-requests-{started_at.strftime('%Y%m%dT%H%M%S_%fZ')}.json"
        self._records: Dict[str, list[dict[str, Any]]] = {}
        self._close_task: Optional[asyncio.Task[None]] = None

    def record_request(
        self, chat_id: str, payload: Dict[str, Any], recorded_at: datetime
    ) -> None:
        """Store a serialized request payload for the given chat identifier."""

        request_payload = dict(payload)
        request_payload["received_at"] = recorded_at.isoformat()
        conversation_entry = {"request": request_payload, "response": None}
        self._records.setdefault(chat_id, []).append(conversation_entry)
        self.last_activity = recorded_at

    def record_response(
        self,
        chat_id: str,
        payload: Optional[Dict[str, Any]],
        recorded_at: datetime,
        status_code: int,
    ) -> None:
        """Attach the assistant response to the latest request for the chat."""

        if payload is None:
            response_payload: Dict[str, Any] = {}
        else:
            response_payload = dict(payload)

        response_payload["responded_at"] = recorded_at.isoformat()
        response_payload["status_code"] = int(status_code)
        history = self._records.setdefault(chat_id, [])

        if history and history[-1].get("response") is None:
            history[-1]["response"] = response_payload
        else:
            history.append({"request": None, "response": response_payload})

        self.last_activity = recorded_at

    def schedule_close_task(self, task: asyncio.Task[None]) -> None:
        """Track the asynchronous task responsible for closing this session."""

        if self._close_task is not None and self._close_task is not task:
            self._close_task.cancel()
        self._close_task = task

    def cancel_close_task(self) -> None:
        """Cancel any pending close task, typically during manual shutdown."""

        if self._close_task is not None:
            self._close_task.cancel()
            self._close_task = None

    def clear_close_task(self, task: Optional[asyncio.Task[None]] = None) -> None:
        """Forget the tracked close task when it completes or is cancelled."""

        if task is None or self._close_task is task:
            self._close_task = None

    def payload(self) -> Dict[str, Any]:
        """Return the JSON-serialisable representation of the session."""

        return {
            "started_at": self.started_at.isoformat(),
            "ended_at": self.last_activity.isoformat(),
            "requests": {
                chat_id: [
                    {
                        "request": dict(entry["request"]) if entry["request"] is not None else None,
                        "response": dict(entry["response"]) if entry["response"] is not None else None,
                    }
                    for entry in entries
                ]
                for chat_id, entries in self._records.items()
            },
        }

    async def write_to_disk(self) -> Path:
        """Persist the collected payload to disk as a JSON document."""

        data = json.dumps(self.payload(), ensure_ascii=False, indent=2)
        await asyncio.to_thread(self.file_path.write_text, data, encoding="utf-8")
        return self.file_path


class RequestLogger:
    """Coordinates request collection and persistence for judge traffic."""

    def __init__(self, directory: Path, inactivity_seconds: float = 60.0) -> None:
        if inactivity_seconds <= 0:
            raise ValueError("'inactivity_seconds' must be greater than zero")

        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)
        self._inactivity_seconds = float(inactivity_seconds)
        self._lock = asyncio.Lock()
        self._session: Optional[_LogSession] = None

    @property
    def directory(self) -> Path:
        """Return the directory where log files are written."""

        return self._directory

    async def log_chat_request(self, request: BaseModel) -> None:
        """Capture judge requests for later inspection when the prefix matches."""

        chat_id = getattr(request, "chat_id", None)
        if not isinstance(chat_id, str) or not chat_id.startswith(_LOGGED_CHAT_PREFIX):
            return

        payload = request.model_dump(mode="json")
        recorded_at = datetime.now(timezone.utc)

        async with self._lock:
            session = self._session
            if session is None:
                session = _LogSession(self._directory, recorded_at)
                self._session = session

            session.record_request(chat_id, payload, recorded_at)
            close_task = asyncio.create_task(self._close_after_timeout(session))
            session.schedule_close_task(close_task)

    async def log_chat_response(
        self,
        chat_id: str,
        response: BaseModel | Mapping[str, Any] | None,
        *,
        status_code: int,
    ) -> None:
        """Capture the assistant response associated with a judge request."""

        if not isinstance(chat_id, str) or not chat_id.startswith(_LOGGED_CHAT_PREFIX):
            return

        if isinstance(response, BaseModel):
            payload: Optional[Dict[str, Any]] = response.model_dump(mode="json")
        elif response is None:
            payload = None
        else:
            payload = dict(response)
        recorded_at = datetime.now(timezone.utc)

        async with self._lock:
            session = self._session
            if session is None:
                session = _LogSession(self._directory, recorded_at)
                self._session = session

            session.record_response(chat_id, payload, recorded_at, status_code)
            close_task = asyncio.create_task(self._close_after_timeout(session))
            session.schedule_close_task(close_task)

    async def aclose(self) -> None:
        """Flush the current session to disk if logging is active."""

        async with self._lock:
            session = self._session
            if session is None:
                return

            self._session = None
            session.cancel_close_task()

        try:
            path = await session.write_to_disk()
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to persist judge request log to disk")
        else:
            logger.info("Persisted judge request log to %s", path)

    async def _close_after_timeout(self, session: _LogSession) -> None:
        """Persist the current session after a period of inactivity."""

        current_task = asyncio.current_task()
        should_persist = False
        try:
            await asyncio.sleep(self._inactivity_seconds)
            async with self._lock:
                if self._session is not session:
                    return

                self._session = None
                should_persist = True
        except asyncio.CancelledError:
            return
        finally:
            if current_task is not None:
                session.clear_close_task(current_task)

        if not should_persist:
            return

        try:
            path = await session.write_to_disk()
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to persist judge request log to disk")
        else:
            logger.info("Persisted judge request log to %s", path)


request_logger = RequestLogger(
    directory=settings.request_log_directory,
    inactivity_seconds=settings.request_log_idle_seconds,
)


__all__ = ["RequestLogger", "request_logger"]

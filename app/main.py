"""Main FastAPI application for the shopping assistant."""

import base64
import binascii
import io
import logging
import zipfile
from decimal import Decimal, InvalidOperation
from typing import Any, List, Literal, Mapping, Optional, Tuple

from pydantic_ai import BinaryContent
from pydantic_ai.usage import UsageLimits
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from tenacity import AsyncRetrying, stop_after_attempt, wait_fixed

from sqlalchemy.ext.asyncio import AsyncSession

from .agent import (
    AgentDependencies,
    RouterDecision,
    get_agent,
    get_router,
    get_scenario4_coordinator,
)
from .agent.image import get_image_agent
from .db import AsyncSessionLocal, get_session
from .logging_utils.judge_requests import request_logger


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


logger = logging.getLogger(__name__)


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


def _decode_image_payload(data: str) -> Tuple[bytes, Optional[str]]:
    """Return raw image bytes and mime type from a base64 payload."""

    payload = data.strip()
    if not payload:
        raise ValueError("Empty image payload.")

    mime_type: Optional[str] = None
    if payload.startswith("data:"):
        header, _, encoded = payload.partition(",")
        if not encoded:
            raise ValueError("Malformed data URL.")
        mime_section = header.split(";", maxsplit=1)[0]
        if mime_section.startswith("data:"):
            mime_candidate = mime_section[5:]
            mime_type = mime_candidate or None
        payload = encoded

    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image data.") from exc

    if not image_bytes:
        raise ValueError("Decoded image payload is empty.")

    return image_bytes, mime_type


async def _run_agent_with_retry(agent: Any, **kwargs: Any) -> Any:
    """Execute the agent while retrying once after a short delay on failure."""

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(2), wait=wait_fixed(0.25), reraise=True
    ):
        with attempt:
            return await agent.run(**kwargs)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest, session: AsyncSession = Depends(get_session)
) -> ChatResponse:
    """Handle chat interactions with the assistant.

    For scenario 0 the handler returns deterministic responses that allow the
    judge to verify that the API is reachable and well-formed.
    """

    async def _safe_log_request() -> None:
        try:
            await request_logger.log_chat_request(request)
        except Exception:  # pragma: no cover - exercised in integration tests
            logger.exception("Failed to log judge chat request")

    async def _safe_log_response(
        payload: ChatResponse | Mapping[str, Any] | None, status_code: int
    ) -> None:
        try:
            await request_logger.log_chat_response(
                request.chat_id, payload, status_code=status_code
            )
        except Exception:  # pragma: no cover - exercised in integration tests
            logger.exception("Failed to log judge chat response")

    async def _finalize(response: ChatResponse, status_code: int = 200) -> ChatResponse:
        await _safe_log_response(response, status_code)
        return response

    await _safe_log_request()

    try:
        if not request.messages:
            return await _finalize(ChatResponse(message="No messages provided."))

        text_segments: List[str] = []
        image_payloads: List[str] = []
        for message in request.messages:
            content = message.content
            stripped = content.strip()
            if not stripped:
                continue
            if message.type == "text":
                text_segments.append(stripped)
            elif message.type == "image":
                image_payloads.append(stripped)

        lower_text_segments = [segment.lower() for segment in text_segments]

        if request.chat_id == "sanity-check-ping" or any(
            segment == "ping" for segment in lower_text_segments
        ):
            return await _finalize(ChatResponse(message="pong"))

        for text in text_segments:
            base_key = _extract_key("return base random key:", text)
            if base_key:
                return await _finalize(ChatResponse(base_random_keys=[base_key]))

        for text in text_segments:
            member_key = _extract_key("return member random key:", text)
            if member_key:
                return await _finalize(ChatResponse(member_random_keys=[member_key]))

        aggregated_prompt = "\n\n".join(text_segments).strip()
        latest_user_message = text_segments[-1] if text_segments else aggregated_prompt

        if image_payloads:
            try:
                image_bytes, mime_type = _decode_image_payload(image_payloads[-1])
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            agent = get_image_agent()
            deps = AgentDependencies(session=session, session_factory=AsyncSessionLocal)

            vision_prompt_text = aggregated_prompt
            if not vision_prompt_text:
                vision_prompt_text = (
                    "کاربر تصویری ارسال کرده است. محتوای تصویر را به اختصار توصیف کن."
                )

            media_type = mime_type or "image/png"
            prompt_segments = [
                vision_prompt_text,
                BinaryContent(data=image_bytes, media_type=media_type),
            ]

            try:
                result = await _run_agent_with_retry(
                    agent,
                    user_prompt=prompt_segments,
                    deps=deps,
                    usage_limits=UsageLimits(
                        request_limit=5,
                        tool_calls_limit=5,
                    ),
                )
            except Exception as exc:  # pragma: no cover - defensive logging path
                raise HTTPException(
                    status_code=500, detail="Vision agent execution failed."
                ) from exc

            reply = result.output.clipped()

            return await _finalize(
                ChatResponse(
                    message=reply.message,
                    base_random_keys=reply.base_random_keys or None,
                    member_random_keys=reply.member_random_keys or None,
                )
            )
        if not aggregated_prompt:
            return await _finalize(
                ChatResponse(message="No textual message found in the request.")
            )

        router_decision = RouterDecision.SINGLE_TURN
        try:
            router = get_router()
            routing_result = await _run_agent_with_retry(
                router, user_prompt=aggregated_prompt
            )
            router_decision = routing_result.output.decision
        except Exception:  # pragma: no cover - defensive logging path
            logger.exception("Routing agent failed; defaulting to single-turn handling.")

        multi_turn_required = router_decision is RouterDecision.MULTI_TURN
        if multi_turn_required:
            coordinator = get_scenario4_coordinator()
            deps = AgentDependencies(session=session, session_factory=AsyncSessionLocal)
            try:
                multi_turn_reply = await coordinator.handle_turn(
                    chat_id=request.chat_id,
                    user_message=latest_user_message,
                    deps=deps,
                    usage_limits=UsageLimits(request_limit=6, tool_calls_limit=10),
                )
            except Exception as exc:  # pragma: no cover - defensive logging path
                raise HTTPException(
                    status_code=500, detail="Scenario 4 coordinator failed."
                ) from exc

            clipped_multi_turn = multi_turn_reply.clipped()
            return await _finalize(
                ChatResponse(
                    message=clipped_multi_turn.message,
                    base_random_keys=clipped_multi_turn.base_random_keys or None,
                    member_random_keys=clipped_multi_turn.member_random_keys or None,
                )
            )

        agent = get_agent()
        deps = AgentDependencies(session=session, session_factory=AsyncSessionLocal)

        try:
            result = await _run_agent_with_retry(
                agent,
                user_prompt=aggregated_prompt,
                deps=deps,
                usage_limits=UsageLimits(
                    request_limit=5,
                    tool_calls_limit=8,
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            raise HTTPException(
                status_code=500, detail="Agent execution failed."
            ) from exc

        reply = result.output.clipped()

        message = reply.message
        if reply.numeric_answer is not None:
            try:
                numeric_value = Decimal(reply.numeric_answer)
            except (
                InvalidOperation,
                TypeError,
            ) as exc:  # pragma: no cover - sanity guard
                raise HTTPException(
                    status_code=500,
                    detail="Agent returned a non-numeric statistic.",
                ) from exc
            if not numeric_value.is_finite():
                raise HTTPException(
                    status_code=500, detail="Agent returned a non-finite statistic."
                )
            message = format(numeric_value.normalize(), "f")

        return await _finalize(
            ChatResponse(
                message=message,
                base_random_keys=reply.base_random_keys or None,
                member_random_keys=reply.member_random_keys or None,
            )
        )
    except HTTPException as exc:
        await _safe_log_response({"detail": exc.detail}, exc.status_code)
        raise
    except Exception as exc:  # pragma: no cover - defensive logging path
        logger.exception("Unhandled error while processing chat request")
        await _safe_log_response(
            {"detail": "Internal server error.", "error": str(exc)}, 500
        )
        raise


@app.get("/download_logs")
async def download_logs(
    include_all: bool = Query(False, alias="all"),
) -> StreamingResponse:
    """Return the latest judge request log (or all logs) as a ZIP archive."""

    await request_logger.aclose()

    log_dir = request_logger.directory
    log_files = sorted(
        log_dir.glob("judge-requests-*.json"), key=lambda path: path.name
    )
    if not log_files:
        raise HTTPException(status_code=404, detail="No judge request logs available.")

    if include_all:
        files_to_archive = log_files
        archive_name = "judge-requests-all.zip"
    else:
        latest_file = log_files[-1]
        files_to_archive = [latest_file]
        archive_name = f"{latest_file.stem}.zip"

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(
        zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for file_path in files_to_archive:
            archive.write(file_path, arcname=file_path.name)

    zip_buffer.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="{archive_name}"'}
    return StreamingResponse(zip_buffer, media_type="application/zip", headers=headers)


async def _shutdown_request_logger() -> None:
    """Ensure any pending judge logs are flushed to disk."""

    await request_logger.aclose()


app.add_event_handler("shutdown", _shutdown_request_logger)


__all__ = [
    "app",
    "chat_endpoint",
    "download_logs",
]

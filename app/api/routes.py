from fastapi import APIRouter, status

from app.agent.assistant import run_chat
from app.config import settings
from app.models.chat import ChatRequest, ChatResponse
from app.services.replay import log_interaction

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    response = await run_chat(request)
    log_interaction(request, response, settings.replay_log_dir)
    return response


@router.get("/health", status_code=status.HTTP_200_OK)
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}

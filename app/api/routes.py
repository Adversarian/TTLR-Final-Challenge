from fastapi import APIRouter, status

from app.models.chat import ChatRequest, ChatResponse
from app.services.scenario_zero import handle_chat

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    return handle_chat(request)


@router.get("/health", status_code=status.HTTP_200_OK)
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}

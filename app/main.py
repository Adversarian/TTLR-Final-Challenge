"""FastAPI application wiring and observability configuration."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from llama_index.core import Settings as LlamaSettings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register

from app import db
from app.api.routes import router as api_router
from app.config import settings

logger = logging.getLogger(__name__)

_INSTRUMENTED = False


def configure_llama_index() -> None:
    if settings.openai_api_key:
        LlamaSettings.llm = LlamaOpenAI(
            model=settings.openai_chat_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    else:
        LlamaSettings.llm = LlamaOpenAI(model=settings.openai_chat_model)

    embed_model_name = settings.openai_embed_model or "text-embedding-3-large"
    LlamaSettings.embed_model = OpenAIEmbedding(
        model=embed_model_name,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def _phoenix_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1/traces"):
        return base
    return f"{base}/v1/traces"


def _instrument_observability() -> None:
    global _INSTRUMENTED
    if _INSTRUMENTED:
        return

    tracer_provider = None
    if settings.phoenix_server_url:
        endpoint = _phoenix_endpoint(settings.phoenix_server_url)
        project_name = settings.phoenix_project or "torob-shopping-assistant"
        try:
            tracer_provider = register(
                project_name=project_name,
                endpoint=endpoint,
            )
        except Exception:  # pragma: no cover - instrumentation should not crash startup
            logger.exception("Failed to register Phoenix exporter", exc_info=True)

    try:
        LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    except Exception:  # pragma: no cover - instrumentation should not crash startup
        logger.exception("Failed to instrument LlamaIndex", exc_info=True)
        return

    _INSTRUMENTED = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.database_url:
        db.init_pool(settings.database_url)

    _instrument_observability()
    try:
        yield
    finally:
        db.close_pool()


def create_app() -> FastAPI:
    configure_llama_index()

    app = FastAPI(title="Torob Shopping Assistant", lifespan=lifespan)
    app.include_router(api_router)
    return app


app = create_app()

from fastapi import FastAPI

from llama_index.core import Settings as LlamaSettings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.observability import ObservabilityConfig
from llama_index.observability.arize_phoenix import ArizePhoenixTracerConfig

from app.api.routes import router as api_router
from app.config import settings
from app import db


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

    tracer_configs = []
    if settings.phoenix_server_url:
        tracer_configs.append(
            ArizePhoenixTracerConfig(
                server_url=settings.phoenix_server_url,
                project_name=settings.phoenix_project,
            )
        )
    if tracer_configs:
        LlamaSettings.observability = ObservabilityConfig(tracer_configs=tracer_configs)


def create_app() -> FastAPI:
    configure_llama_index()

    app = FastAPI(title="Torob Shopping Assistant")
    app.include_router(api_router)

    @app.on_event("startup")
    def _startup() -> None:
        if settings.database_url:
            db.init_pool(settings.database_url)

    @app.on_event("shutdown")
    def _shutdown() -> None:
        db.close_pool()

    return app


app = create_app()

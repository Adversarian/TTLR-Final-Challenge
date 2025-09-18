from fastapi import FastAPI
import logfire

from app.api.routes import router as api_router
from app.config import settings
from app import db


def create_app() -> FastAPI:
    configure_kwargs = {}
    if settings.logfire_api_key:
        configure_kwargs["api_key"] = settings.logfire_api_key
    logfire.configure(**configure_kwargs)
    try:
        logfire.instrument_psycopg()
    except Exception as exc:  # pragma: no cover - best effort
        logfire.warning("psycopg_instrumentation_failed", error=str(exc))
    try:
        logfire.instrument_openai()
    except Exception as exc:  # pragma: no cover - best effort
        logfire.warning("openai_instrumentation_failed", error=str(exc))

    app = FastAPI(title="Torob Shopping Assistant")
    try:
        logfire.instrument_fastapi(app)
    except Exception as exc:  # pragma: no cover - best effort
        logfire.warning("fastapi_instrumentation_failed", error=str(exc))
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

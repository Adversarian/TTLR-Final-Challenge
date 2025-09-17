from fastapi import FastAPI

from app.api.routes import router as api_router
from app.config import settings
from app import db


def create_app() -> FastAPI:
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

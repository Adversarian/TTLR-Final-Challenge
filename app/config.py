from dataclasses import dataclass
import os


@dataclass
class Settings:
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "info"
    database_url: str | None = None
    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_reasoning_model: str | None = None
    openai_embed_model: str | None = None
    openai_base_url: str | None = None
    openai_reasoning_effort: str | None = None
    logfire_api_key: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            app_host=os.getenv("APP_HOST", "0.0.0.0"),
            app_port=int(os.getenv("APP_PORT", "8000")),
            log_level=os.getenv("LOG_LEVEL", "info"),
            database_url=os.getenv("DATABASE_URL"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            openai_reasoning_model=os.getenv("OPENAI_REASONING_MODEL"),
            openai_embed_model=os.getenv("OPENAI_EMBED_MODEL"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            openai_reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT"),
            logfire_api_key=os.getenv("LOGFIRE_API_KEY"),
        )


settings = Settings.from_env()

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
    openai_embed_model: str | None = None
    openai_base_url: str | None = None
    phoenix_server_url: str | None = None
    phoenix_project: str | None = None
    replay_log_dir: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            app_host=os.getenv("APP_HOST", "0.0.0.0"),
            app_port=int(os.getenv("APP_PORT", "8000")),
            log_level=os.getenv("LOG_LEVEL", "info"),
            database_url=os.getenv("DATABASE_URL"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            openai_embed_model=os.getenv("OPENAI_EMBED_MODEL"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            phoenix_server_url=os.getenv("PHOENIX_SERVER_URL"),
            phoenix_project=os.getenv("PHOENIX_PROJECT_NAME"),
            replay_log_dir=os.getenv("REPLAY_LOG_DIR"),
        )


settings = Settings.from_env()

from dataclasses import dataclass
import os


@dataclass
class Settings:
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "info"
    database_url: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            app_host=os.getenv("APP_HOST", "0.0.0.0"),
            app_port=int(os.getenv("APP_PORT", "8000")),
            log_level=os.getenv("LOG_LEVEL", "info"),
            database_url=os.getenv("DATABASE_URL"),
        )


settings = Settings.from_env()

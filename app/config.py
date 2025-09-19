"""Application configuration management utilities."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Runtime configuration for the shopping assistant service."""

    database_url: str = Field(..., alias="DATABASE_URL")
    primary_model: str = Field(..., alias="PRIMARY_MODEL")
    log_directory: Path = Field(Path("./logs"), alias="LOG_DIRECTORY")
    conversation_log_name: str = Field("conversations.jsonl", alias="CONVERSATION_LOG_NAME")
    logfire_api_key: Optional[str] = Field(default=None, alias="LOGFIRE_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    @property
    def conversation_log_path(self) -> Path:
        """Absolute path to the file where chat transcripts are recorded."""

        return self.log_directory / self.conversation_log_name


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    settings = Settings()
    settings.log_directory.mkdir(parents=True, exist_ok=True)
    return settings

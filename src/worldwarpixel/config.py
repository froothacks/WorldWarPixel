from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: SecretStr = SecretStr("")
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    max_concurrency: int = 10
    request_timeout: float = 60.0
    event_log_path: str = "events.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()

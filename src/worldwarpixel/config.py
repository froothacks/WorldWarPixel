from functools import lru_cache
from urllib.parse import urlparse

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: SecretStr = SecretStr("")
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    agent_model: str | None = None
    narrator_model: str | None = None
    agent_vision: bool = True
    modal_key: SecretStr = SecretStr("")
    modal_secret: SecretStr = SecretStr("")
    max_concurrency: int = 10
    request_timeout: float = 60.0
    event_log_path: str = "events.db"
    max_active_runs: int = 4

    def default_headers(self) -> dict[str, str]:
        """Modal proxy-auth headers, only when the base URL actually points at a Modal host."""
        key, secret = self.modal_key.get_secret_value(), self.modal_secret.get_secret_value()
        host = urlparse(self.openai_base_url or "").hostname or ""
        is_modal = host == "modal.run" or host.endswith((".modal.run", ".modal.direct"))
        return {"Modal-Key": key, "Modal-Secret": secret} if key and secret and is_modal else {}


@lru_cache
def get_settings() -> Settings:
    return Settings()

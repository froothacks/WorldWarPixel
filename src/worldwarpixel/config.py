from functools import lru_cache
from urllib.parse import urlparse

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: SecretStr = SecretStr("")
    """Optional bearer key for OpenAI-style endpoints; never sent to Modal hosts."""
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    agent_model: str | None = None
    narrator_model: str | None = None
    agent_vision: bool = True
    agent_max_tokens: int = Field(default=700, ge=1)
    """Max output tokens per agent model call; reasoning models need more to reach the tool call."""
    narrator_max_tokens: int = Field(default=700, ge=1)
    """Max output tokens per narrator update; reasoning models need more to finish the JSON."""
    modal_key: SecretStr = SecretStr("")
    modal_secret: SecretStr = SecretStr("")
    god_token: SecretStr = SecretStr("")
    """When set, god-mode endpoints require this value in the X-God-Token header."""
    max_concurrency: int = 10
    request_timeout: float = 60.0
    event_log_path: str = "events.db"
    max_active_runs: int = 4

    @property
    def is_modal_endpoint(self) -> bool:
        host = urlparse(self.openai_base_url or "").hostname or ""
        return host == "modal.run" or host.endswith((".modal.run", ".modal.direct"))

    def api_key(self) -> str:
        """Bearer key for the model endpoint; 'none' (ignored) on Modal, which uses headers."""
        return (
            "none" if self.is_modal_endpoint else self.openai_api_key.get_secret_value() or "none"
        )

    def default_headers(self) -> dict[str, str]:
        """Modal proxy-auth headers, only when the base URL actually points at a Modal host."""
        key, secret = self.modal_key.get_secret_value(), self.modal_secret.get_secret_value()
        return (
            {"Modal-Key": key, "Modal-Secret": secret}
            if key and secret and self.is_modal_endpoint
            else {}
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()

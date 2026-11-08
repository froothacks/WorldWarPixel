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
    agent_vision: bool = True
    agent_max_tokens: int = Field(default=700, ge=1)
    """Max output tokens per agent model call; reasoning models need more to reach the tool call."""
    narrator_max_tokens: int = Field(default=700, ge=1)
    """Max output tokens per narrator update; reasoning models need more to finish the JSON."""
    modal_key: SecretStr = SecretStr("")
    modal_secret: SecretStr = SecretStr("")
    qwen_base_url: str = "https://advait-m--ep-world-war-pixel-server.us-west.modal.direct/v1"
    """Endpoint of the 'qwen' model preset (see models.py); override for another deployment."""
    deepseek_base_url: str = (
        "https://advait-m--ep-deepseek-v4-1-flash-server.us-west.modal.direct/v1"
    )
    """Endpoint of the 'deepseek' model preset (see models.py); override for another deployment."""
    glm_base_url: str = "https://chryztoph--ep-glm-5-3-flash-nvfp4-server.us-west.modal.direct/v1"
    """Endpoint of the 'glm' model preset (see models.py); override for another deployment."""
    god_token: SecretStr = SecretStr("")
    """When set, god-mode endpoints require this value in the X-God-Token header."""
    device_tokens: SecretStr = SecretStr("")
    """Society screens: "token:society,token:society". Empty = devices are open (dev only)."""
    whisper_model: str = "base"
    """faster-whisper model size used to transcribe society-screen voice clips."""
    murmur_condense: bool = False
    """Rephrase voice transcripts into a rumour with the server's chat model."""
    voice_dump_dir: str = ""
    """When set, every received voice clip is saved there as a WAV (for debugging devices)."""
    max_concurrency: int = 10
    request_timeout: float = 60.0
    event_log_path: str = "events.db"
    max_active_runs: int = 4

    @property
    def is_modal_endpoint(self) -> bool:
        return is_modal_url(self.openai_base_url)

    def api_key(self) -> str:
        """Bearer key for the model endpoint; 'none' (ignored) on Modal, which uses headers."""
        return (
            "none" if self.is_modal_endpoint else self.openai_api_key.get_secret_value() or "none"
        )

    def modal_headers(self) -> dict[str, str]:
        """Modal proxy-auth headers whenever the keys exist, whatever the chat endpoint is."""
        key, secret = self.modal_key.get_secret_value(), self.modal_secret.get_secret_value()
        return {"Modal-Key": key, "Modal-Secret": secret} if key and secret else {}

    def modal_headers_for(self, url: str | None) -> dict[str, str]:
        """Modal headers only when the destination is a Modal host, so an overridden
        preset base URL can never receive the Modal credentials."""
        return self.modal_headers() if is_modal_url(url) else {}

    def default_headers(self) -> dict[str, str]:
        """Modal proxy-auth headers, only when the base URL actually points at a Modal host."""
        return self.modal_headers() if self.is_modal_endpoint else {}


def is_modal_url(url: str | None) -> bool:
    host = urlparse(url or "").hostname or ""
    return host == "modal.run" or host.endswith((".modal.run", ".modal.direct"))


@lru_cache
def get_settings() -> Settings:
    return Settings()

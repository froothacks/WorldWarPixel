from functools import lru_cache
from urllib.parse import urlparse

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: SecretStr = SecretStr("")
    """Optional bearer key for OpenAI-style endpoints; never sent to Modal hosts."""
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    agent_model: str | None = None
    agent_endpoints: str = ""
    """Extra agent models selectable per run: "model=base_url,model=base_url"."""
    agent_text_only: str = ""
    """Comma-separated agent model ids that get no canvas image (images make GLM ramble)."""
    agent_reasoning_effort: str = ""
    """reasoning_effort for AGENT_ENDPOINTS models, replacing enable_thinking=False ("low")."""
    narrator_model: str | None = None
    agent_vision: bool = True
    modal_key: SecretStr = SecretStr("")
    modal_secret: SecretStr = SecretStr("")
    god_token: SecretStr = SecretStr("")
    """When set, god-mode endpoints require this value in the X-God-Token header."""
    device_tokens: SecretStr = SecretStr("")
    """Society screens: "token:society,token:society". Empty = devices are open (dev only)."""
    whisper_model: str = "base"
    murmur_condense: bool = False
    voice_dump_dir: str = ""
    """When set, every received voice clip is saved there as a WAV (for debugging devices)."""
    """Rephrase voice transcripts into a rumour with the narrator model (small models mangle it)."""
    """faster-whisper model size used to transcribe society-screen voice clips."""
    max_concurrency: int = 10
    request_timeout: float = 60.0
    event_log_path: str = "events.db"
    max_active_runs: int = 4

    @property
    def is_modal_endpoint(self) -> bool:
        return is_modal_url(self.openai_base_url)

    def agent_endpoint_map(self) -> dict[str, str]:
        """Parsed AGENT_ENDPOINTS: model id -> base URL, in the configured order."""
        out: dict[str, str] = {}
        for item in self.agent_endpoints.split(","):
            model, sep, url = item.partition("=")
            if sep and model.strip() and url.strip():
                out[model.strip()] = url.strip()
        return out

    def api_key(self, base_url: str | None = None) -> str:
        """Bearer key for a model endpoint; 'none' (ignored) on Modal, which uses headers."""
        url = base_url if base_url is not None else self.openai_base_url
        return "none" if is_modal_url(url) else self.openai_api_key.get_secret_value() or "none"

    def default_headers(self, base_url: str | None = None) -> dict[str, str]:
        """Modal proxy-auth headers, only when the base URL actually points at a Modal host."""
        url = base_url if base_url is not None else self.openai_base_url
        key, secret = self.modal_key.get_secret_value(), self.modal_secret.get_secret_value()
        return (
            {"Modal-Key": key, "Modal-Secret": secret}
            if key and secret and is_modal_url(url)
            else {}
        )


def is_modal_url(url: str | None) -> bool:
    host = urlparse(url or "").hostname or ""
    return host == "modal.run" or host.endswith((".modal.run", ".modal.direct"))


@lru_cache
def get_settings() -> Settings:
    return Settings()

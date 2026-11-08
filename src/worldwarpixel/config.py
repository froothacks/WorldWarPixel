import os
from functools import lru_cache
from urllib.parse import urlparse

from dotenv import dotenv_values
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
    timelapse_dir: str = "timelapses"
    """Where rendered run videos ({run_id}.mp4 + .json) are written; on Modal a Volume path."""
    timelapse_auto: bool = True
    """Render a video automatically when a run is stopped."""
    timelapse_autosave_interval: float = Field(default=30.0, ge=0, le=3600)
    """Seconds between autosaves of the in-progress video while a run is live (0 disables).
    Each autosave overwrites the same {run_id}.mp4, so the archive keeps one growing video per
    run instead of a prefix per snapshot."""
    timelapse_duration: float = Field(default=20.0, ge=2, le=120)
    """Length of the rendered timelapse in seconds (plus short holds at both ends)."""
    timelapse_s3_bucket: str | None = None
    """Optional S3-compatible bucket to mirror videos into (AWS creds from the environment)."""
    timelapse_s3_endpoint: str | None = None
    """Custom S3 endpoint (https://storage.googleapis.com for GCS, or R2); None for AWS."""
    timelapse_s3_prefix: str = "timelapses/"
    ffmpeg_path: str = "ffmpeg"
    god_token: SecretStr = SecretStr("")
    """When set, god-mode endpoints require this value in the X-God-Token header."""
    tts_api_key: SecretStr = SecretStr("")
    """Speech panel voices: an OpenAI key, or set TTS_BASE_URL for a compatible speech service."""
    tts_base_url: str | None = None
    tts_model: str = "gpt-4o-mini-tts"
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

    def secret(self, name: str) -> str:
        """A credential named by a model endpoint (models.py), from os.environ or .env.

        Endpoint secrets are not Settings fields, so pydantic never maps them out of .env;
        read the file directly. An env var that is set (even to "") wins over .env, matching
        pydantic-settings precedence."""
        if name in os.environ:
            return os.environ[name]
        return dotenv_values(".env").get(name) or ""

    def default_headers(self) -> dict[str, str]:
        """Modal proxy-auth headers, only when the base URL actually points at a Modal host."""
        return self.modal_headers() if self.is_modal_endpoint else {}


def is_modal_url(url: str | None) -> bool:
    host = urlparse(url or "").hostname or ""
    return host == "modal.run" or host.endswith((".modal.run", ".modal.direct"))


@lru_cache
def get_settings() -> Settings:
    return Settings()

"""Selectable agent models: endpoint, auth style and per-model API quirks.

The start screen picks one per run. The quirks are load-bearing:

- ``gpt-6-luna`` on api.openai.com rejects ``max_tokens`` (wants
  ``max_completion_tokens``), rejects our Qwen ``chat_template_kwargs`` hint with a 400,
  and rejects function tools unless ``reasoning_effort="none"``.
- The Modal vLLM endpoints want the classic ``max_tokens`` plus the
  ``chat_template_kwargs`` thinking hint, and DeepSeek needs a much larger output budget
  because it reasons before calling tools.
"""

from dataclasses import dataclass
from typing import Literal

from worldwarpixel.config import Settings

TokenParam = Literal["max_tokens", "max_completion_tokens"]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Everything needed to talk to one model, plus the quirks of its endpoint."""

    id: str
    """Selector value sent as RunConfig.model."""
    label: str
    """Human-readable name for the start screen."""
    model: str
    """Model id sent to the endpoint."""
    base_url: str | None
    auth: Literal["modal", "bearer"]
    """modal: Modal-Key/Modal-Secret headers; bearer: OPENAI_API_KEY as the bearer token."""
    token_param: TokenParam = "max_tokens"
    chat_template_kwargs: bool = True
    """Send extra_body chat_template_kwargs.enable_thinking=False (vLLM/Qwen only)."""
    reasoning_effort: str | None = None
    """OpenAI reasoning_effort; gpt-6-luna requires "none" whenever tools are attached."""
    max_tokens: int | None = None
    """Default agent output budget; None falls back to Settings.agent_max_tokens."""
    narrator_max_tokens: int | None = None
    """Default narrator output budget; None falls back to Settings.narrator_max_tokens."""


def model_registry(settings: Settings) -> list[ModelSpec]:
    """The models offered in the UI, best first."""
    return [
        ModelSpec(
            id="gpt-6-luna",
            label="GPT-6 Luna (OpenAI)",
            model="gpt-6-luna",
            base_url="https://api.openai.com/v1",
            auth="bearer",
            token_param="max_completion_tokens",
            chat_template_kwargs=False,
            reasoning_effort="none",
            max_tokens=3000,
            narrator_max_tokens=2000,
        ),
        ModelSpec(
            id="qwen",
            label="Qwen (SANIC Modal)",
            model="world-war-pixel",
            base_url=settings.qwen_base_url,
            auth="modal",
        ),
        ModelSpec(
            id="deepseek",
            label="DeepSeek V4.1 Flash (Modal)",
            model="deepseek-ai/DeepSeek-V4.1-Flash",
            base_url=settings.deepseek_base_url,
            auth="modal",
            max_tokens=3000,
            narrator_max_tokens=3000,
        ),
    ]


def find_model(settings: Settings, model_id: str) -> ModelSpec | None:
    return next((m for m in model_registry(settings) if m.id == model_id), None)


def model_available(settings: Settings, spec: ModelSpec) -> bool:
    """Whether this server holds the credentials the model's auth style needs."""
    if spec.auth == "modal":
        return bool(
            settings.modal_key.get_secret_value() and settings.modal_secret.get_secret_value()
        )
    return bool(settings.openai_api_key.get_secret_value())

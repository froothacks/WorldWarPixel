"""Selectable agent models: the endpoints serving each one, their secrets, and API quirks.

The start screen picks one preset per run. A preset may list several endpoints serving the
same model (e.g. GLM deployed in three Modal workspaces); the server multiplexes a run's
agents across every endpoint whose secrets it holds, see routing.py.

The quirks are load-bearing:

- ``gpt-6-luna`` on api.openai.com rejects ``max_tokens`` (wants
  ``max_completion_tokens``), rejects our Qwen ``chat_template_kwargs`` hint with a 400,
  and rejects function tools unless ``reasoning_effort="none"``.
- The Modal vLLM endpoints want the classic ``max_tokens`` plus the
  ``chat_template_kwargs`` thinking hint, and DeepSeek needs a much larger output budget
  because it reasons before calling tools.
- GLM needs ``reasoning_effort="low"`` without the thinking hint (with the hint it ignores
  the effort and thinks until the cap), rate-limits (429) large fast runs per endpoint, and
  gets no canvas image.
"""

from dataclasses import dataclass
from typing import Literal

from worldwarpixel.config import Settings
from worldwarpixel.routing import Endpoint, endpoint_headers, modal_endpoint

TokenParam = Literal["max_tokens", "max_completion_tokens"]

OPENAI = Endpoint(
    name="openai",
    base_url="https://api.openai.com/v1",
    model="gpt-6-luna",
    auth="bearer",
    secrets=("OPENAI_API_KEY",),
)
# Modal workspaces and the env vars holding a proxy-auth token pair for each. A pair only
# works for endpoints in its own workspace (other workspaces answer 401).
ADVAIT = ""  # advait-m: MODAL_KEY / MODAL_SECRET
CHRYZTOPH = "CHRYZTOPH_"
MEGANWCHUN = "MEGANWCHUN_"

GLM_MODEL = "nvidia/GLM-5.3-Flash-NVFP4"
GLM_SERVER = "ep-glm-5-3-flash-nvfp4-server"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One selectable model: where it is served, plus the quirks of its API."""

    id: str
    """Selector value sent as RunConfig.model."""
    label: str
    """Human-readable name for the start screen."""
    endpoints: tuple[Endpoint, ...]
    """Servers for this model; calls are spread over the ones with credentials present."""
    aliases: tuple[str, ...] = ()
    """Former preset ids that now resolve to this one (stored runs, saved UI choices)."""
    token_param: TokenParam = "max_tokens"
    chat_template_kwargs: bool = True
    """Send extra_body chat_template_kwargs.enable_thinking=False (vLLM/Qwen only)."""
    reasoning_effort: str | None = None
    """OpenAI reasoning_effort; gpt-6-luna requires "none" whenever tools are attached."""
    max_tokens: int | None = None
    """Default agent output budget; None falls back to Settings.agent_max_tokens."""
    narrator_max_tokens: int | None = None
    """Default narrator output budget; None falls back to Settings.narrator_max_tokens."""
    vision: bool = True
    """Show the model the canvas image (GLM reasons for minutes over images, so it gets text)."""
    concurrency: int | None = None
    """Cap on simultaneous model calls per preset; None means no preset cap."""

    @property
    def model(self) -> str:
        return self.endpoints[0].model


def model_registry() -> list[ModelSpec]:
    """The models offered in the UI, best first."""
    return [
        ModelSpec(
            id="gpt-6-luna",
            label="GPT-6 Luna",
            endpoints=(OPENAI,),
            token_param="max_completion_tokens",
            chat_template_kwargs=False,
            reasoning_effort="none",
            max_tokens=3000,
            narrator_max_tokens=2000,
        ),
        ModelSpec(
            id="qwen",
            label="Qwen 0.8B",
            endpoints=(
                modal_endpoint(
                    "advait-m", "ep-world-war-pixel-server", "world-war-pixel", secret_prefix=ADVAIT
                ),
                modal_endpoint("chryztoph", "ep-qwen-server", "qwen", secret_prefix=CHRYZTOPH),
            ),
        ),
        ModelSpec(
            id="deepseek",
            label="DeepSeek V4.1 Flash",
            endpoints=(
                modal_endpoint(
                    "advait-m",
                    "ep-deepseek-v4-1-flash-server",
                    "deepseek-ai/DeepSeek-V4.1-Flash",
                    secret_prefix=ADVAIT,
                ),
            ),
            max_tokens=3000,
            narrator_max_tokens=3000,
            # The shared endpoint 429s above ~16 concurrent calls.
            concurrency=16,
        ),
        ModelSpec(
            id="glm",
            label="GLM 5.3 Flash NVFP4",
            endpoints=(
                modal_endpoint("advait-m", GLM_SERVER, GLM_MODEL, secret_prefix=ADVAIT),
                modal_endpoint("chryztoph", GLM_SERVER, GLM_MODEL, secret_prefix=CHRYZTOPH),
                modal_endpoint("meganwchun", GLM_SERVER, GLM_MODEL, secret_prefix=MEGANWCHUN),
            ),
            aliases=("glm-5.3-flash", "glm-megan"),
            # Reasons (reasoning_content) before the tool call: at the default effort it burns a
            # 3000-token budget thinking and never calls a tool, so keep effort low and leave room
            # for a full place_pixels batch.
            chat_template_kwargs=False,
            reasoning_effort="low",
            vision=False,
            max_tokens=8000,
            narrator_max_tokens=8000,
        ),
    ]


def find_model(model_id: str) -> ModelSpec | None:
    return next((m for m in model_registry() if model_id in (m.id, *m.aliases)), None)


def available_endpoints(settings: Settings, spec: ModelSpec) -> list[Endpoint]:
    """The preset's endpoints whose secrets this server holds."""
    return [e for e in spec.endpoints if endpoint_headers(e, settings.secret) is not None]


def model_available(settings: Settings, spec: ModelSpec) -> bool:
    return bool(available_endpoints(settings, spec))

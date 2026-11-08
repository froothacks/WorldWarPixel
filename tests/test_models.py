import pytest

from worldwarpixel.config import Settings
from worldwarpixel.models import available_endpoints, find_model, model_available, model_registry

SECRETS = (
    "OPENAI_API_KEY",
    "MODAL_KEY",
    "MODAL_SECRET",
    "CHRYZTOPH_MODAL_KEY",
    "CHRYZTOPH_MODAL_SECRET",
    "MEGANWCHUN_MODAL_KEY",
    "MEGANWCHUN_MODAL_SECRET",
)


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    for var in SECRETS:
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    return Settings()


def test_registry_has_the_known_models_with_their_quirks() -> None:
    specs = {m.id: m for m in model_registry()}
    assert set(specs) == {"gpt-6-luna", "qwen", "deepseek", "glm"}

    luna = specs["gpt-6-luna"]
    (openai,) = luna.endpoints
    assert openai.base_url == "https://api.openai.com/v1"
    assert openai.auth == "bearer" and openai.secrets == ("OPENAI_API_KEY",)
    assert luna.token_param == "max_completion_tokens"
    assert not luna.chat_template_kwargs
    assert luna.reasoning_effort == "none"
    assert luna.max_tokens == 3000

    qwen = specs["qwen"]
    assert qwen.model == "world-war-pixel"
    assert qwen.token_param == "max_tokens"
    assert qwen.chat_template_kwargs
    assert qwen.reasoning_effort is None
    assert qwen.max_tokens is None  # falls back to AGENT_MAX_TOKENS

    deepseek = specs["deepseek"]
    assert deepseek.max_tokens == 3000
    # The shared DeepSeek endpoint 429s above ~16 concurrent calls; the preset caps it.
    assert deepseek.concurrency == 16
    assert specs["gpt-6-luna"].concurrency is None
    assert specs["qwen"].concurrency is None

    glm = specs["glm"]
    assert glm.model == "nvidia/GLM-5.3-Flash-NVFP4"
    assert not glm.vision and not glm.chat_template_kwargs
    # Default effort reasons through the whole budget and never calls a tool.
    assert glm.reasoning_effort == "low"
    assert glm.max_tokens == 8000 and glm.narrator_max_tokens == 8000


def test_modal_endpoints_pair_secrets_with_their_workspace() -> None:
    for spec in model_registry():
        assert len({e.name for e in spec.endpoints}) == len(spec.endpoints)
        for e in spec.endpoints:
            if e.auth != "modal":
                continue
            # A Modal token only works in its own workspace (hostname prefix).
            assert e.base_url.startswith(f"https://{e.name}--")
            prefix = {"advait-m": ""}.get(e.name, e.name.upper().replace("-", "_") + "_")
            assert e.secrets == (f"{prefix}MODAL_KEY", f"{prefix}MODAL_SECRET")
    glm = find_model("glm")
    assert glm is not None
    assert {e.name for e in glm.endpoints} == {"advait-m", "chryztoph", "meganwchun"}
    assert {e.model for e in glm.endpoints} == {glm.model}
    qwen = find_model("qwen")
    assert qwen is not None
    assert [e.name for e in qwen.endpoints] == ["advait-m", "chryztoph"]
    assert [e.secrets for e in qwen.endpoints] == [
        ("MODAL_KEY", "MODAL_SECRET"),
        ("CHRYZTOPH_MODAL_KEY", "CHRYZTOPH_MODAL_SECRET"),
    ]
    assert [e.model for e in qwen.endpoints] == ["world-war-pixel", "qwen"]


def test_availability_follows_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    assert not any(model_available(settings, m) for m in model_registry())

    settings = _settings(monkeypatch, **{v: "x" for v in SECRETS})
    assert all(model_available(settings, m) for m in model_registry())

    settings = _settings(monkeypatch, MODAL_KEY="k", MODAL_SECRET="s")
    assert [m.id for m in model_registry() if model_available(settings, m)] == [
        "qwen",
        "deepseek",
        "glm",
    ]


def test_pool_uses_only_endpoints_with_complete_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    glm = find_model("glm")
    assert glm is not None
    settings = _settings(
        monkeypatch,
        CHRYZTOPH_MODAL_KEY="ck",
        CHRYZTOPH_MODAL_SECRET="cs",
        MEGANWCHUN_MODAL_KEY="mk",  # secret half missing
    )
    assert [e.name for e in available_endpoints(settings, glm)] == ["chryztoph"]


def test_find_model_resolves_legacy_ids() -> None:
    assert find_model("nope") is None
    assert find_model("qwen-c") is None
    for legacy in ("glm-5.3-flash", "glm-megan"):
        spec = find_model(legacy)
        assert spec is not None and spec.id == "glm"

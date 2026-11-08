import pytest

from worldwarpixel.config import Settings
from worldwarpixel.models import find_model, model_available, model_registry


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    for var in (
        "OPENAI_API_KEY",
        "MODAL_KEY",
        "MODAL_SECRET",
        "CHRYZTOPH_MODAL_KEY",
        "CHRYZTOPH_MODAL_SECRET",
    ):
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    return Settings()


def test_registry_has_the_known_models_with_their_quirks() -> None:
    specs = {m.id: m for m in model_registry(Settings())}
    assert set(specs) == {"gpt-6-luna", "qwen", "deepseek", "qwen-c", "glm-5.3-flash"}

    luna = specs["gpt-6-luna"]
    assert luna.base_url == "https://api.openai.com/v1"
    assert luna.auth == "bearer"
    assert luna.token_param == "max_completion_tokens"
    assert not luna.chat_template_kwargs
    assert luna.reasoning_effort == "none"
    assert luna.max_tokens == 3000

    qwen = specs["qwen"]
    assert qwen.auth == "modal"
    assert qwen.token_param == "max_tokens"
    assert qwen.chat_template_kwargs
    assert qwen.reasoning_effort is None
    assert qwen.max_tokens is None  # falls back to AGENT_MAX_TOKENS

    glm = specs["glm-5.3-flash"]
    assert glm.auth == "modal" and not glm.vision
    assert not glm.chat_template_kwargs and glm.reasoning_effort == "low"


def test_modal_base_urls_are_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, QWEN_BASE_URL="https://example.modal.direct/v1")
    qwen = find_model(settings, "qwen")
    assert qwen is not None and qwen.base_url == "https://example.modal.direct/v1"


def test_availability_follows_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    specs = model_registry(settings)
    assert not any(model_available(settings, m) for m in specs)

    settings = _settings(
        monkeypatch,
        MODAL_KEY="k",
        MODAL_SECRET="s",
        OPENAI_API_KEY="sk-test",
        CHRYZTOPH_MODAL_KEY="ck",
        CHRYZTOPH_MODAL_SECRET="cs",
    )
    specs = model_registry(settings)
    assert all(model_available(settings, m) for m in specs)

    settings = _settings(monkeypatch, MODAL_KEY="k", MODAL_SECRET="s")
    specs = model_registry(settings)
    assert [model_available(settings, m) for m in specs] == [
        False,
        True,
        True,
        False,  # chryztoph workspace presets need their own credential pair
        False,
    ]


def test_modal_preset_overridden_to_non_modal_url_needs_no_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch, QWEN_BASE_URL="http://localhost:8000/v1")
    qwen = find_model(settings, "qwen")
    assert qwen is not None
    assert model_available(settings, qwen)


def test_find_model_unknown_id() -> None:
    assert find_model(Settings(), "nope") is None

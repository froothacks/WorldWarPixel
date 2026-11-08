import pytest
from pydantic import ValidationError

from worldwarpixel.config import Settings


def test_modal_headers_only_for_modal_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODAL_KEY", "k")
    monkeypatch.setenv("MODAL_SECRET", "s")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://x--y.us-west.modal.direct/v1")
    assert Settings().default_headers() == {"Modal-Key": "k", "Modal-Secret": "s"}
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    assert Settings().default_headers() == {}
    monkeypatch.delenv("OPENAI_BASE_URL")
    assert Settings().default_headers() == {}


def test_modal_headers_do_not_depend_on_the_chat_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    # Modal model presets must authenticate even when OPENAI_BASE_URL points at api.openai.com.
    monkeypatch.setenv("MODAL_KEY", "k")
    monkeypatch.setenv("MODAL_SECRET", "s")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    settings = Settings()
    assert settings.default_headers() == {}
    assert settings.modal_headers() == {"Modal-Key": "k", "Modal-Secret": "s"}
    monkeypatch.delenv("MODAL_KEY")
    assert Settings().modal_headers() == {}


def test_modal_headers_for_only_on_modal_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    # An overridden preset base URL must never receive Modal credentials.
    monkeypatch.setenv("MODAL_KEY", "k")
    monkeypatch.setenv("MODAL_SECRET", "s")
    settings = Settings()
    assert settings.modal_headers_for("https://advait-m--ep-x.us-west.modal.direct/v1") == {
        "Modal-Key": "k",
        "Modal-Secret": "s",
    }
    assert settings.modal_headers_for("https://evil.example.com/v1") == {}
    assert settings.modal_headers_for(None) == {}


@pytest.mark.parametrize("var", ["AGENT_MAX_TOKENS", "NARRATOR_MAX_TOKENS"])
def test_max_tokens_must_be_positive(monkeypatch: pytest.MonkeyPatch, var: str) -> None:
    monkeypatch.setenv(var, "0")
    with pytest.raises(ValidationError):
        Settings()

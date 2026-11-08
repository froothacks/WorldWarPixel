from pathlib import Path

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


def test_secret_reads_env_then_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Endpoint secrets are not Settings fields, so they only reach us via os.environ or .env.
    monkeypatch.delenv("CHRYZTOPH_MODAL_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("CHRYZTOPH_MODAL_KEY=dk\n")
    assert Settings().secret("CHRYZTOPH_MODAL_KEY") == "dk"
    assert Settings().secret("NOPE") == ""
    monkeypatch.setenv("CHRYZTOPH_MODAL_KEY", "ek")
    assert Settings().secret("CHRYZTOPH_MODAL_KEY") == "ek"
    # An env var set to "" (operator cleared it) wins over .env.
    monkeypatch.setenv("CHRYZTOPH_MODAL_KEY", "")
    assert Settings().secret("CHRYZTOPH_MODAL_KEY") == ""


@pytest.mark.parametrize("var", ["AGENT_MAX_TOKENS", "NARRATOR_MAX_TOKENS"])
def test_max_tokens_must_be_positive(monkeypatch: pytest.MonkeyPatch, var: str) -> None:
    monkeypatch.setenv(var, "0")
    with pytest.raises(ValidationError):
        Settings()

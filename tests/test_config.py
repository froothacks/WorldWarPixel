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


def test_agent_max_tokens_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MAX_TOKENS", "0")
    with pytest.raises(ValidationError):
        Settings()

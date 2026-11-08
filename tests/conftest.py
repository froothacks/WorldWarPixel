import pytest

from worldwarpixel.config import get_settings


@pytest.fixture(autouse=True)
def _isolate_timelapses(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests that stop runs must not render videos into the repo's timelapse directory."""
    monkeypatch.setenv("TIMELAPSE_AUTO", "false")
    monkeypatch.setenv("TIMELAPSE_DIR", str(tmp_path_factory.mktemp("timelapses")))
    get_settings.cache_clear()

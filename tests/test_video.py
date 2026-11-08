import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from worldwarpixel.app import app
from worldwarpixel.config import get_settings
from worldwarpixel.sim.agents import RandomWalkPolicy
from worldwarpixel.sim.events import EventLog
from worldwarpixel.sim.run import Run, RunConfig
from worldwarpixel.sim.video import FrameComposer, VideoError, ffmpeg_available, load_timeline
from worldwarpixel.videos import VideoStore

CONFIG = RunConfig(
    width=48, height=30, agents=4, cooldown=0.01, jitter=0, pixels_per_turn=4, seed=5,
    policy="random", narrator_interval=5,
)  # fmt: skip
needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")


def _run(log: EventLog, run_id: str = "v", seconds: float = 0.3) -> Run:
    async def main() -> Run:
        run = Run(run_id, CONFIG, log, RandomWalkPolicy(seed=5))
        await run.start()
        await asyncio.sleep(seconds)
        run.narrator.history.append({"kind": "update", "headline": "Early scramble"})
        run.record("narrator.update", None, {"kind": "update", "headline": "Early scramble"})
        run.record("narrator.update", None, {"kind": "answer", "headline": "ignored"})
        await run.stop()
        return run

    return asyncio.run(main())


def test_load_timeline_collects_posts_headlines_and_span() -> None:
    log = EventLog()
    run = _run(log)
    tl = load_timeline(log, "v")
    assert tl is not None
    assert tl.config["width"] == 48 and len(tl.agents) == 4
    assert tl.pixels == run.stats["pixels"] > 0
    assert len(tl.posts) == run.stats["posts"]
    assert tl.headlines == [(tl.headlines[0][0], "Early scramble")]
    assert tl.t0 is not None and tl.t1 is not None and tl.t0 <= tl.t1
    assert sum(tl.pixels_by_country.values()) == tl.pixels
    assert load_timeline(log, "nope") is None


def test_frame_composer_has_even_dimensions_and_integer_scale() -> None:
    log = EventLog()
    _run(log)
    tl = load_timeline(log, "v")
    assert tl is not None
    composer = FrameComposer(tl, 48, 30)
    assert composer.scale == 15 and composer.canvas_w == 720
    assert composer.w % 2 == 0 and composer.h % 2 == 0
    frame = composer.compose(bytes(48 * 30), tl.t0 or 0, 0, tl.posts[:3], "Early scramble")
    assert len(frame) == composer.w * composer.h * 3


@needs_ffmpeg
def test_render_video_writes_mp4_and_metadata(tmp_path: Path) -> None:
    log = EventLog()
    run = _run(log)
    store = VideoStore(tmp_path, duration=2)
    meta = store.render_sync(log, "v")
    assert store.video_path("v").exists() and store.video_path("v").stat().st_size > 1000
    assert store.video_path("v").read_bytes()[4:8] == b"ftyp"
    assert meta["pixels"] == run.stats["pixels"] and meta["posts"] == run.stats["posts"]
    assert meta["frames"] == 2 * 30 + 15 + 60 and meta["duration_s"] == 4.5
    assert meta["narrator_headlines"] == ["Early scramble"]
    assert json.loads(store.meta_path("v").read_text())["run_id"] == "v"
    assert [m["run_id"] for m in store.list()] == ["v"]
    assert store.meta("missing") is None


def test_narrator_headline_after_last_pixel_extends_the_span() -> None:
    log = EventLog()
    _run(log)
    tl = load_timeline(log, "v")
    assert tl is not None
    last_activity = max(e.ts for e in log.read("v", limit=100_000) if e.kind in ("pixel", "post"))
    assert tl.t1 is not None and tl.t1 >= tl.headlines[-1][0] > last_activity


def test_reader_opens_a_second_connection_for_file_logs(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "e.db")
    _run(log)
    reader = log.reader()
    assert reader is not log and len(reader.read("v", limit=10)) == 10
    reader.close()
    assert EventLog().reader() is not None


def test_render_without_activity_raises(tmp_path: Path) -> None:
    log = EventLog()
    log.append("empty", 1.0, "run.created", None, {"config": CONFIG.model_dump(), "agents": []})
    with pytest.raises(VideoError, match="no canvas or board activity"):
        VideoStore(tmp_path).render_sync(log, "empty")


@needs_ffmpeg
def test_s3_mirror_uploads_video_and_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploads: list[tuple[str, str, str]] = []

    class FakeS3:
        def upload_file(self, path: str, bucket: str, key: str, ExtraArgs: dict[str, str]) -> None:
            uploads.append((bucket, key, ExtraArgs["ContentType"]))

    monkeypatch.setattr(VideoStore, "_client", lambda self: FakeS3())
    log = EventLog()
    _run(log)
    store = VideoStore(tmp_path, s3_bucket="clips", s3_prefix="wwp/", duration=2)
    meta = store.render_sync(log, "v")
    assert meta["s3"] == {"bucket": "clips", "key": "wwp/v.mp4", "endpoint": "aws"}
    assert uploads == [
        ("clips", "wwp/v.mp4", "video/mp4"),
        ("clips", "wwp/v.json", "application/json"),
    ]


@needs_ffmpeg
def test_s3_failure_keeps_the_local_video_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenS3:
        def upload_file(self, *a: Any, **kw: Any) -> None:
            raise RuntimeError("bucket unreachable")

    monkeypatch.setattr(VideoStore, "_client", lambda self: BrokenS3())
    log = EventLog()
    _run(log)
    store = VideoStore(tmp_path, s3_bucket="clips", duration=2)
    meta = store.render_sync(log, "v")
    assert "s3" not in meta and store.meta("v") is not None and len(store.list()) == 1
    monkeypatch.setattr(VideoStore, "_client", lambda self: FakeS3Ok())
    assert store.mirror_sync("v") is True
    assert store.meta("v")["s3"]["key"] == "timelapses/v.mp4"  # type: ignore[index]


class FakeS3Ok:
    def upload_file(self, *a: Any, **kw: Any) -> None:
        pass


@needs_ffmpeg
def test_scheduled_render_after_stop_is_fresh_and_drained(tmp_path: Path) -> None:
    async def main() -> None:
        log = EventLog()
        run = Run("v", CONFIG, log, RandomWalkPolicy(seed=5))
        await run.start()
        await asyncio.sleep(0.2)
        store = VideoStore(tmp_path, duration=2)
        early = asyncio.ensure_future(store.render(log, "v"))
        await asyncio.sleep(0.05)
        await run.stop()
        store.schedule(log, "v")
        snapshot = await early
        await store.drain()
        final = store.meta("v")
        assert final is not None and final["pixels"] == run.stats["pixels"] >= snapshot["pixels"]
        assert final["rendered_at"] > snapshot["rendered_at"]

    asyncio.run(main())


def test_autosave_rerenders_the_same_path_until_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    async def main() -> None:
        log = EventLog()
        run = Run("v", CONFIG, log, RandomWalkPolicy(seed=5))
        await run.start()
        await asyncio.sleep(0.1)
        await run.stop()
        store = VideoStore(tmp_path, duration=2, autosave_interval=0.05)

        def fake_render(log_: EventLog, run_id: str) -> dict[str, Any]:
            calls.append(run_id)
            return {"run_id": run_id, "bytes": 1}

        monkeypatch.setattr(store, "render_sync", fake_render)
        store.start_autosave(log, "v")
        await asyncio.sleep(0.2)
        store.stop_autosave("v")
        count = len(calls)
        await asyncio.sleep(0.12)
        assert count >= 2 and len(calls) == count  # cancelled: no more renders
        assert set(calls) == {"v"}  # every autosave targets the same video path

    asyncio.run(main())


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("EVENT_LOG_PATH", ":memory:")
    monkeypatch.setenv("TIMELAPSE_DIR", str(tmp_path))
    monkeypatch.setenv("TIMELAPSE_DURATION", "2")
    monkeypatch.setenv("TIMELAPSE_AUTO", "true")
    get_settings.cache_clear()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


@needs_ffmpeg
def test_video_endpoints_render_on_stop_and_on_demand(client: TestClient) -> None:
    body: dict[str, Any] = {**CONFIG.model_dump(), "cooldown": 0.01}
    run_id = client.post("/runs", json=body).json()["id"]
    assert client.get(f"/runs/{run_id}/video").status_code in (404, 202)
    assert client.get(f"/runs/{run_id}/video/meta").json()["rendered"] is False
    import time

    time.sleep(0.3)
    assert client.post(f"/runs/{run_id}/stop").json()["state"] == "stopped"
    manager = client.app.state.runs  # type: ignore[attr-defined]
    for _ in range(100):  # the auto-render runs in the background
        if manager.videos.meta(run_id) is not None:
            break
        time.sleep(0.1)
    meta = client.get(f"/runs/{run_id}/video/meta").json()
    assert meta["rendered"] is True and meta["run_id"] == run_id
    video = client.get(f"/runs/{run_id}/video")
    assert video.status_code == 200 and video.headers["content-type"] == "video/mp4"
    assert video.content[4:8] == b"ftyp"
    assert client.post(f"/runs/{run_id}/video").status_code == 201
    assert [m["run_id"] for m in client.get("/videos").json()] == [run_id]
    assert client.get("/runs/nope/video").status_code == 404

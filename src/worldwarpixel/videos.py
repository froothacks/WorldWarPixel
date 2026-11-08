"""Where rendered timelapse videos go: a local directory (on Modal, the persistent Volume) and,
when configured, a mirror in an S3-compatible bucket so the archive can be browsed and curated
outside the app. Each ``{run_id}.mp4`` has a ``{run_id}.json`` sidecar with the run's metadata."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
from pathlib import Path
from typing import Any

from worldwarpixel.sim.events import EventStore, open_event_log
from worldwarpixel.sim.video import VideoError, render_video

log = logging.getLogger(__name__)


class VideoStore:
    def __init__(
        self,
        directory: str | Path,
        *,
        s3_bucket: str | None = None,
        s3_endpoint: str | None = None,
        s3_prefix: str = "timelapses/",
        duration: float | None = None,
        ffmpeg: str = "ffmpeg",
        autosave_interval: float = 0.0,
    ) -> None:
        self.dir = Path(directory)
        self.s3_bucket = s3_bucket or None
        self.s3_endpoint = s3_endpoint or None
        self.s3_prefix = s3_prefix
        self.duration = duration
        self.ffmpeg = ffmpeg
        self.autosave_interval = autosave_interval
        self._rendering: dict[str, asyncio.Task[dict[str, Any]]] = {}
        self._pending: set[asyncio.Task[None]] = set()
        self._autosave: dict[str, asyncio.Task[None]] = {}
        self._encoder = threading.Semaphore(1)  # one ffmpeg at a time, whoever asks

    # -- paths -----------------------------------------------------------------------------

    def video_path(self, run_id: str) -> Path:
        return self.dir / f"{run_id}.mp4"

    def meta_path(self, run_id: str) -> Path:
        return self.dir / f"{run_id}.json"

    def meta(self, run_id: str) -> dict[str, Any] | None:
        p = self.meta_path(run_id)
        if not p.exists() or not self.video_path(run_id).exists():
            return None
        data: dict[str, Any] = json.loads(p.read_text())
        return data

    def list(self) -> list[dict[str, Any]]:
        if not self.dir.exists():
            return []
        rows = [m for p in self.dir.glob("*.json") if (m := self.meta(p.stem)) is not None]
        return sorted(rows, key=lambda m: m.get("rendered_at", 0), reverse=True)

    def rendering(self, run_id: str) -> bool:
        task = self._rendering.get(run_id)
        return task is not None and not task.done()

    # -- rendering -------------------------------------------------------------------------

    def render_sync(self, log_: EventStore, run_id: str) -> dict[str, Any]:
        """Render, write the sidecar and mirror to S3; blocking, run it in a thread."""
        reader = log_.reader()
        try:
            with self._encoder:
                meta = render_video(
                    reader,
                    run_id,
                    self.video_path(run_id),
                    duration=self.duration,
                    ffmpeg=self.ffmpeg,
                )
        finally:
            if reader is not log_:
                reader.close()
        self.meta_path(run_id).write_text(json.dumps(meta, indent=1))
        if self.s3_bucket:
            self.mirror_sync(run_id, meta)
        return meta

    def mirror_sync(self, run_id: str, meta: dict[str, Any] | None = None) -> bool:
        """Upload an already rendered video + sidecar to the bucket, recording the key in the
        sidecar; failures are logged so the local copy stays usable. Returns success."""
        meta = meta if meta is not None else self.meta(run_id)
        if meta is None:
            return False
        sidecar = self.meta_path(run_id)
        try:
            meta["s3"] = self._upload(run_id)
            sidecar.write_text(json.dumps(meta, indent=1))
            self._put(sidecar, f"{run_id}.json", "application/json")
            return True
        except Exception:
            meta.pop("s3", None)
            log.exception("S3 mirror failed for %s; the local copy is kept", run_id)
            return False

    async def render(self, log_: EventStore, run_id: str, *, fresh: bool = False) -> dict[str, Any]:
        """Render in a worker thread; concurrent calls for the same run share one render.
        ``fresh`` waits for an in-flight render to finish and then renders again, so the
        result includes everything logged up to now (used when the run has just stopped)."""
        task = self._rendering.get(run_id)
        if task is not None and not task.done() and fresh:
            with contextlib.suppress(Exception):
                await task
            task = None
        if task is None or task.done():
            task = asyncio.create_task(asyncio.to_thread(self.render_sync, log_, run_id))
            self._rendering[run_id] = task
        return await task

    def schedule(self, log_: EventStore, run_id: str) -> None:
        """Fire-and-forget render (used when a run stops); failures are logged, not raised."""

        async def go() -> None:
            try:
                meta = await self.render(log_, run_id, fresh=True)
                log.info("rendered timelapse for %s (%s bytes)", run_id, meta["bytes"])
            except VideoError as e:
                log.info("no timelapse for %s: %s", run_id, e)
            except Exception:
                log.exception("timelapse render failed for %s", run_id)

        task = asyncio.get_running_loop().create_task(go())
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    # -- autosave ----------------------------------------------------------------------------

    def start_autosave(self, log_: EventStore, run_id: str) -> None:
        """Re-render the run's video every ``autosave_interval`` seconds, overwriting the same
        {run_id}.mp4/.json (and S3 keys), so a live run's archive entry is always current."""
        if self.autosave_interval <= 0 or run_id in self._autosave:
            return

        async def loop() -> None:
            while True:
                await asyncio.sleep(self.autosave_interval)
                try:
                    # fresh=False: an in-flight render (e.g. an on-demand request) is awaited
                    # instead of duplicated; the encoder semaphore serializes ffmpeg anyway.
                    meta = await self.render(log_, run_id)
                    log.info("autosaved timelapse for %s (%s bytes)", run_id, meta["bytes"])
                except VideoError as e:
                    log.info("no timelapse autosave for %s: %s", run_id, e)
                except Exception:
                    log.exception("timelapse autosave failed for %s", run_id)

        task = asyncio.get_running_loop().create_task(loop())
        self._autosave[run_id] = task
        task.add_done_callback(lambda _: self._autosave.pop(run_id, None))

    def stop_autosave(self, run_id: str) -> None:
        task = self._autosave.pop(run_id, None)
        if task is not None:
            task.cancel()

    def stop_all_autosaves(self) -> None:
        for run_id in list(self._autosave):
            self.stop_autosave(run_id)

    async def drain(self, timeout: float = 120.0) -> None:
        """Wait for scheduled renders (called at shutdown, before the event log is closed)."""
        if self._pending:
            await asyncio.wait(set(self._pending), timeout=timeout)

    # -- S3 mirror -------------------------------------------------------------------------

    def _client(self) -> Any:
        import boto3  # type: ignore[import-untyped]
        from botocore.config import Config  # type: ignore[import-untyped]

        # GCS and R2 reject the default CRC checksums / aws-chunked uploads newer boto3 sends.
        config = Config(
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        )
        return boto3.client("s3", endpoint_url=self.s3_endpoint, config=config)

    def _put(self, path: Path, name: str, content_type: str) -> str:
        key = f"{self.s3_prefix}{name}"
        self._client().upload_file(
            str(path), self.s3_bucket, key, ExtraArgs={"ContentType": content_type}
        )
        return key

    def _upload(self, run_id: str) -> dict[str, str]:
        key = self._put(self.video_path(run_id), f"{run_id}.mp4", "video/mp4")
        return {"bucket": str(self.s3_bucket), "key": key, "endpoint": self.s3_endpoint or "aws"}


def main(argv: list[str] | None = None) -> int:
    """``python -m worldwarpixel.videos [run_id ...]``: render missing videos for logged runs."""
    import sys

    from worldwarpixel.config import get_settings

    settings = get_settings()
    store = VideoStore(
        settings.timelapse_dir,
        s3_bucket=settings.timelapse_s3_bucket,
        s3_endpoint=settings.timelapse_s3_endpoint,
        s3_prefix=settings.timelapse_s3_prefix,
        duration=settings.timelapse_duration,
        ffmpeg=settings.ffmpeg_path,
    )
    events = open_event_log(settings)
    try:
        wanted = argv if argv is not None else sys.argv[1:]
        run_ids = wanted or [r for r in events.run_ids() if store.meta(r) is None]
        failures = 0
        if store.s3_bucket and not wanted:  # repair failed S3 mirrors without re-rendering
            for m in store.list():
                if "s3" not in m and store.mirror_sync(m["run_id"], m):
                    print(f"{m['run_id']}: mirrored existing video")
        for run_id in run_ids:
            try:
                meta = store.render_sync(events, run_id)
                print(f"{run_id}: {meta['file']} ({meta['bytes'] / 1e6:.1f} MB)")
            except VideoError as e:
                failures += 1
                print(f"{run_id}: skipped ({e})")
        return 1 if failures and wanted else 0
    finally:
        events.close()


if __name__ == "__main__":
    raise SystemExit(main())

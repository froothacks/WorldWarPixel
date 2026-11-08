import { useEffect, useMemo, useRef, useState } from "react";
import { InlineError } from "./Errors";

interface Frame {
  ts: number;
  deltas: Uint32Array; // pairs packed as index * 16 + colour would overflow; keep two arrays
  colors: Uint8Array;
}

interface TimelapseData {
  width: number;
  height: number;
  total: number;
  frames: Frame[];
}

interface Props {
  runId: string;
  palette: string[];
  onClose: () => void;
}

const VIDEO_TYPES = ["video/webm;codecs=vp9", "video/webm", "video/mp4"];

/** First container this browser can record a canvas into (Safari has no webm), or null. */
function videoType(): string | null {
  if (
    typeof MediaRecorder === "undefined" ||
    !("captureStream" in HTMLCanvasElement.prototype)
  )
    return null;
  return VIDEO_TYPES.find((t) => MediaRecorder.isTypeSupported(t)) ?? null;
}

/** Replays the run's canvas history from the /timelapse endpoint with a scrubber and webm export. */
export default function Timelapse({ runId, palette, onClose }: Props) {
  const [data, setData] = useState<TimelapseData | null>(null);
  const [loaded, setLoaded] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pos, setPos] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [fps, setFps] = useState(12);
  const [recording, setRecording] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bufRef = useRef<Uint8Array | null>(null);
  const builtRef = useRef(-1);
  const recRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const exportType = useMemo(() => videoType(), []);

  const stopCapture = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  };

  useEffect(
    () => () => {
      if (recRef.current?.state === "recording") recRef.current.stop();
      stopCapture();
    },
    [],
  );

  useEffect(() => {
    let cancelled = false;
    let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
    let streamData: TimelapseData | null = null;
    let buffer = new Uint8Array(64 * 1024);
    let used = 0;
    let offset = 0;
    let published = 0;
    let lastPublishedAt = 0;

    bufRef.current = null;
    builtRef.current = -1;

    const publishProgress = (force = false) => {
      if (!streamData) return;
      const count = streamData.frames.length;
      const now = performance.now();
      if (
        count > published &&
        (force || published === 0 || now - lastPublishedAt >= 200)
      ) {
        published = count;
        lastPublishedAt = now;
        setLoaded(count);
      }
    };

    const consume = (bytes: Uint8Array) => {
      if (buffer.length - used < bytes.length) {
        const next = new Uint8Array(
          Math.max(used + bytes.length, buffer.length * 2),
        );
        next.set(buffer.subarray(0, used));
        buffer = next;
      }
      buffer.set(bytes, used);
      used += bytes.length;

      if (!streamData && used >= 6) {
        const header = new DataView(buffer.buffer, buffer.byteOffset, used);
        streamData = {
          total: header.getUint16(0, true),
          width: header.getUint16(2, true),
          height: header.getUint16(4, true),
          frames: [],
        };
        setData(streamData);
        offset = 6;
      }

      if (streamData) {
        while (used - offset >= 12) {
          const view = new DataView(
            buffer.buffer,
            buffer.byteOffset + offset,
            used - offset,
          );
          const ts = view.getFloat64(0, true);
          const count = view.getUint32(8, true);
          const frameBytes = 12 + count * 5;
          if (used - offset < frameBytes) break;

          const deltas = new Uint32Array(count);
          const colors = new Uint8Array(count);
          let frameOffset = 12;
          for (let i = 0; i < count; i++) {
            deltas[i] = view.getUint32(frameOffset, true);
            colors[i] = view.getUint8(frameOffset + 4);
            frameOffset += 5;
          }
          streamData.frames.push({ ts, deltas, colors });
          offset += frameBytes;
        }
      }

      if (offset > 0) {
        buffer.copyWithin(0, offset, used);
        used -= offset;
        offset = 0;
      }
      publishProgress();
    };

    const progressTimer = window.setInterval(() => publishProgress(), 200);
    const load = async () => {
      setData(null);
      setLoaded(0);
      setLoading(true);
      setError(null);
      setPos(0);
      setPlaying(true);
      try {
        const response = await fetch(`/runs/${runId}/timelapse`);
        if (!response.ok) throw new Error(response.statusText);
        if (cancelled) return;
        reader = response.body!.getReader();
        while (!cancelled) {
          const { value, done } = await reader.read();
          if (done) break;
          if (value) consume(value);
        }
        if (cancelled) return;
        if (!streamData) throw new Error("timelapse response is missing its header");
        if (used !== 0 || streamData.frames.length !== streamData.total) {
          throw new Error("timelapse response ended before all frames arrived");
        }
        publishProgress(true);
        window.clearInterval(progressTimer);
        setLoading(false);
        if (streamData.frames.length === 0) setPlaying(false);
      } catch (e) {
        window.clearInterval(progressTimer);
        if (cancelled) return;
        if (streamData) setLoaded(streamData.frames.length);
        setError((e as Error).message);
        setLoading(false);
        if (streamData?.frames.length === 0) setPlaying(false);
      }
    };

    void load();
    return () => {
      cancelled = true;
      window.clearInterval(progressTimer);
      if (reader) void reader.cancel().catch(() => undefined);
    };
  }, [runId]);

  const rgb = useMemo(
    () =>
      palette.map((hex) => {
        const v = parseInt(hex.slice(1), 16);
        return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
      }),
    [palette],
  );

  // Rebuild the pixel buffer up to `pos` (incrementally when moving forward) and draw it.
  useEffect(() => {
    const el = canvasRef.current;
    if (!data || !el) return;
    const { width, height, frames } = data;
    const available = Math.min(loaded, frames.length);
    if (
      !bufRef.current ||
      bufRef.current.length !== width * height ||
      pos < builtRef.current
    ) {
      bufRef.current = new Uint8Array(width * height);
      builtRef.current = -1;
    }
    const buf = bufRef.current;
    for (let f = builtRef.current + 1; f <= pos && f < available; f++) {
      const { deltas, colors } = frames[f];
      for (let i = 0; i < deltas.length; i++) buf[deltas[i]] = colors[i];
    }
    builtRef.current = Math.min(pos, available - 1);
    el.width = width;
    el.height = height;
    const ctx = el.getContext("2d")!;
    const img = ctx.createImageData(width, height);
    for (let i = 0; i < buf.length; i++) {
      const [r, g, b] = rgb[buf[i]] ?? [255, 255, 255];
      img.data[i * 4] = r;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [data, loaded, pos, rgb]);

  const posRef = useRef(pos);
  useEffect(() => {
    posRef.current = pos;
  }, [pos]);

  useEffect(() => {
    if (!playing || !data || loaded === 0) return;
    const t = window.setInterval(() => {
      if (posRef.current + 1 < loaded) {
        setPos(posRef.current + 1);
        return;
      }
      if (loading) return;
      setPlaying(false);
      const rec = recRef.current;
      if (rec)
        window.setTimeout(() => rec.state === "recording" && rec.stop(), 500);
    }, 1000 / fps);
    return () => window.clearInterval(t);
  }, [playing, data, loaded, loading, fps]);

  const exportComplete =
    data !== null && !loading && data.frames.length === data.total;

  const exportVideo = () => {
    const el = canvasRef.current;
    if (!el || !exportComplete || !data || recording || !exportType) return;
    const stream = el.captureStream(fps);
    let rec: MediaRecorder;
    try {
      rec = new MediaRecorder(stream, { mimeType: exportType });
    } catch (e) {
      stream.getTracks().forEach((t) => t.stop());
      setError(`video export failed: ${(e as Error).message}`);
      return;
    }
    const chunks: Blob[] = [];
    rec.ondataavailable = (e) => chunks.push(e.data);
    rec.onstop = () => {
      recRef.current = null;
      stopCapture();
      setRecording(false);
      if (!canvasRef.current?.isConnected) return;
      const url = URL.createObjectURL(new Blob(chunks, { type: exportType }));
      const a = document.createElement("a");
      a.href = url;
      a.download = `worldwarpixel-${runId}.${exportType.startsWith("video/mp4") ? "mp4" : "webm"}`;
      a.click();
      // Revoking synchronously can cancel the download in Firefox/Safari.
      window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
    };
    streamRef.current = stream;
    recRef.current = rec;
    setRecording(true);
    bufRef.current = null;
    setPos(0);
    rec.start();
    setPlaying(true);
  };

  const frame = data?.frames[pos];
  const t0 = data?.frames[0]?.ts ?? 0;
  const elapsed = frame ? Math.round(frame.ts - t0) : 0;

  return (
    <div className="modal" onClick={onClose}>
      <div
        className="modal-body timelapse"
        onClick={(e) => e.stopPropagation()}
      >
        <header>
          <strong>Timelapse</strong>
          <span className="muted small">
            {error ? (
              <InlineError message={error} />
            ) : data
                ? loading
                  ? `loading ${loaded}/${data.total} frames`
                  : `${data.total} frames · ${Math.round((data.frames.at(-1)?.ts ?? t0) - t0)} s of history`
                : "loading…"}
          </span>
          <span className="spacer" />
          <button className="ghost" onClick={onClose}>
            close
          </button>
        </header>
        <canvas
          ref={canvasRef}
          className="timelapse-canvas"
          // Fit by height too so the controls below stay on screen for tall canvases.
          style={
            data && data.height > 0
              ? {
                  width: `min(80vw, 1024px, calc((92vh - 110px) * ${data.width / data.height}))`,
                }
              : undefined
          }
        />
        <div className="timelapse-controls">
          <button
            onClick={() => setPlaying(!playing)}
            disabled={!data || recording}
          >
            {playing ? "Pause" : "Play"}
          </button>
          <button onClick={() => setPos(0)} disabled={!data || recording}>
            restart
          </button>
          <input
            type="range"
            disabled={!data || loaded === 0 || recording}
            min={0}
            max={Math.max(0, loaded - 1)}
            value={Math.min(pos, Math.max(0, loaded - 1))}
            onChange={(e) => {
              setPlaying(false);
              setPos(+e.target.value);
            }}
          />
          <span className="small muted">
            {loaded ? pos + 1 : 0}/{loaded} · +{elapsed}s
          </span>
          <label className="small">
            fps
            <select value={fps} onChange={(e) => setFps(+e.target.value)}>
              {[4, 8, 12, 24, 30, 60].map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <button
            onClick={exportVideo}
            disabled={!exportComplete || recording || !exportType}
            title={
              exportType
                ? "Plays from the start while recording, then downloads the video"
                : "This browser cannot record a canvas to video"
            }
          >
            {recording
              ? "Recording…"
              : `Export .${exportType?.startsWith("video/mp4") ? "mp4" : "webm"}`}
          </button>
        </div>
      </div>
    </div>
  );
}

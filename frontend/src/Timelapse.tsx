import { useEffect, useMemo, useRef, useState } from "react";

interface Frame {
  ts: number;
  deltas: Uint32Array; // pairs packed as index * 16 + colour would overflow; keep two arrays
  colors: Uint8Array;
}

interface Props {
  runId: string;
  palette: string[];
  onClose: () => void;
}

function parse(buf: ArrayBuffer): {
  width: number;
  height: number;
  frames: Frame[];
} {
  const view = new DataView(buf);
  const n = view.getUint16(0, true);
  const width = view.getUint16(2, true);
  const height = view.getUint16(4, true);
  let off = 6;
  const frames: Frame[] = [];
  for (let f = 0; f < n; f++) {
    const ts = view.getFloat64(off, true);
    const count = view.getUint32(off + 8, true);
    off += 12;
    const deltas = new Uint32Array(count);
    const colors = new Uint8Array(count);
    for (let i = 0; i < count; i++) {
      deltas[i] = view.getUint32(off, true);
      colors[i] = view.getUint8(off + 4);
      off += 5;
    }
    frames.push({ ts, deltas, colors });
  }
  return { width, height, frames };
}

/** Replays the run's canvas history from the /timelapse endpoint with a scrubber and webm export. */
export default function Timelapse({ runId, palette, onClose }: Props) {
  const [data, setData] = useState<{
    width: number;
    height: number;
    frames: Frame[];
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pos, setPos] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [fps, setFps] = useState(12);
  const [recording, setRecording] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bufRef = useRef<Uint8Array | null>(null);
  const builtRef = useRef(-1);
  const recRef = useRef<MediaRecorder | null>(null);

  useEffect(() => () => recRef.current?.stop(), []);

  useEffect(() => {
    fetch(`/runs/${runId}/timelapse?frames=240`)
      .then((r) =>
        r.ok ? r.arrayBuffer() : Promise.reject(new Error(r.statusText)),
      )
      .then((buf) => setData(parse(buf)))
      .catch((e: Error) => setError(e.message));
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
    if (
      !bufRef.current ||
      bufRef.current.length !== width * height ||
      pos < builtRef.current
    ) {
      bufRef.current = new Uint8Array(width * height);
      builtRef.current = -1;
    }
    const buf = bufRef.current;
    for (let f = builtRef.current + 1; f <= pos && f < frames.length; f++) {
      const { deltas, colors } = frames[f];
      for (let i = 0; i < deltas.length; i++) buf[deltas[i]] = colors[i];
    }
    builtRef.current = Math.min(pos, frames.length - 1);
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
  }, [data, pos, rgb]);

  useEffect(() => {
    if (!playing || !data) return;
    const t = window.setInterval(() => {
      setPos((p) => {
        if (p + 1 >= data.frames.length) {
          setPlaying(false);
          if (recRef.current)
            window.setTimeout(() => recRef.current?.stop(), 500);
          return p;
        }
        return p + 1;
      });
    }, 1000 / fps);
    return () => window.clearInterval(t);
  }, [playing, data, fps]);

  const exportVideo = () => {
    const el = canvasRef.current;
    if (!el || !data || recording) return;
    const stream = el.captureStream(fps);
    const rec = new MediaRecorder(stream, { mimeType: "video/webm" });
    const chunks: Blob[] = [];
    rec.ondataavailable = (e) => chunks.push(e.data);
    rec.onstop = () => {
      recRef.current = null;
      setRecording(false);
      if (!canvasRef.current?.isConnected) return;
      const url = URL.createObjectURL(new Blob(chunks, { type: "video/webm" }));
      const a = document.createElement("a");
      a.href = url;
      a.download = `worldwarpixel-${runId}.webm`;
      a.click();
      URL.revokeObjectURL(url);
    };
    recRef.current = rec;
    setRecording(true);
    builtRef.current = -1;
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
            {data
              ? `${data.frames.length} frames · ${Math.round((data.frames.at(-1)?.ts ?? t0) - t0)} s of history`
              : (error ?? "loading…")}
          </span>
          <span className="spacer" />
          <button className="ghost" onClick={onClose}>
            close
          </button>
        </header>
        <canvas ref={canvasRef} className="timelapse-canvas" />
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
            disabled={!data || recording}
            min={0}
            max={Math.max(0, (data?.frames.length ?? 1) - 1)}
            value={pos}
            onChange={(e) => {
              setPlaying(false);
              setPos(+e.target.value);
            }}
          />
          <span className="small muted">
            {pos + 1}/{data?.frames.length ?? 0} · +{elapsed}s
          </span>
          <label className="small">
            fps
            <select value={fps} onChange={(e) => setFps(+e.target.value)}>
              {[4, 8, 12, 24, 30].map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <button
            onClick={exportVideo}
            disabled={!data || recording}
            title="Plays from the start while recording, then downloads a .webm"
          >
            {recording ? "Recording…" : "Export .webm"}
          </button>
        </div>
      </div>
    </div>
  );
}

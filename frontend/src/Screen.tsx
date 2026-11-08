import { useCallback, useEffect, useRef, useState } from "react";

interface SocietyState {
  id: number;
  name: string;
  color: string;
  members: number;
  joined?: number;
  left?: number;
}

interface DeviceState {
  run: string | null;
  society?: SocietyState;
  societies?: SocietyState[];
  threads?: { title: string; author: string; replies: number }[];
  murmur?: string | null;
  voice: { status: string; transcript: string | null; error: string | null };
}

const THUMB_W = 128;
const THUMB_H = 80;
const MAX_SECONDS = 20;

/** Browser stand-in for one ESP32 society screen: same /devices protocol, laptop mic. */
export default function Screen({ society }: { society: number }) {
  const params = new URLSearchParams(location.search);
  const token = params.get("token");
  const run = params.get("run");
  const auth = `society=${society}${token ? `&token=${encodeURIComponent(token)}` : ""}${run ? `&run=${encodeURIComponent(run)}` : ""}`;
  const [state, setState] = useState<DeviceState | null>(null);
  const [palette, setPalette] = useState<string[]>([]);
  const [conn, setConn] = useState("connecting");
  const [toast, setToast] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);
  const [text, setText] = useState("");
  const ws = useRef<WebSocket | null>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const mic = useRef<{ stop: () => void } | null>(null);
  const held = useRef(false);

  useEffect(() => {
    let closed = false;
    let retry: number | undefined;
    const connect = () => {
      const sock = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/devices/ws?${auth}`);
      sock.binaryType = "arraybuffer";
      ws.current = sock;
      sock.onopen = () => setConn("live");
      sock.onmessage = (m) => {
        const msg = JSON.parse(m.data as string);
        if (msg.type === "hello") setPalette(msg.palette);
        else if (msg.type === "state") setState(msg);
        else if (msg.type === "migration") {
          setToast(msg.to === society ? `${msg.agent} joined` : `${msg.agent} left`);
          window.setTimeout(() => setToast(null), 3000);
        }
      };
      sock.onclose = (e) => {
        setConn(e.code === 4401 ? "bad token" : "offline");
        if (!closed && e.code !== 4401) retry = window.setTimeout(connect, 2000);
      };
    };
    connect();
    return () => {
      closed = true;
      window.clearTimeout(retry);
      ws.current?.close();
    };
  }, [auth, society]);

  const runId = state?.run;
  useEffect(() => {
    if (!runId || palette.length === 0) return;
    const draw = async () => {
      const r = await fetch(`/devices/canvas?w=${THUMB_W}&h=${THUMB_H}&${auth}`);
      if (!r.ok || !canvas.current) return;
      const px = new Uint8Array(await r.arrayBuffer());
      const ctx = canvas.current.getContext("2d")!;
      const img = ctx.createImageData(THUMB_W, THUMB_H);
      const rgb = palette.map((h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)));
      px.forEach((c, i) => {
        const [r0, g, b] = rgb[c] ?? [0, 0, 0];
        img.data.set([r0, g, b, 255], i * 4);
      });
      ctx.putImageData(img, 0, 0);
    };
    draw().catch(() => undefined);
    const t = window.setInterval(() => draw().catch(() => undefined), 3000);
    return () => window.clearInterval(t);
  }, [runId, palette, auth]);

  const startTalk = useCallback(async () => {
    const sock = ws.current;
    if (!sock || sock.readyState !== WebSocket.OPEN || mic.current) return;
    held.current = true;
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true } });
    } catch {
      setToast("no microphone access");
      return;
    }
    if (!held.current) {
      stream.getTracks().forEach((t) => t.stop());
      return;
    }
    const ctx = new AudioContext();
    const src = ctx.createMediaStreamSource(stream);
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    sock.send(JSON.stringify({ type: "voice.start", rate: ctx.sampleRate }));
    proc.onaudioprocess = (e) => {
      const f = e.inputBuffer.getChannelData(0);
      const out = new Int16Array(f.length);
      for (let i = 0; i < f.length; i++) out[i] = Math.max(-1, Math.min(1, f[i])) * 0x7fff;
      sock.send(out.buffer);
    };
    src.connect(proc);
    proc.connect(ctx.destination);
    const limit = window.setTimeout(() => mic.current?.stop(), MAX_SECONDS * 1000);
    mic.current = {
      stop: () => {
        window.clearTimeout(limit);
        proc.disconnect();
        src.disconnect();
        stream.getTracks().forEach((t) => t.stop());
        ctx.close().catch(() => undefined);
        sock.send(JSON.stringify({ type: "voice.end" }));
        mic.current = null;
        setRecording(false);
      },
    };
    setRecording(true);
  }, []);

  const stopTalk = () => {
    held.current = false;
    mic.current?.stop();
  };

  const say = async () => {
    if (!text.trim()) return;
    const r = await fetch(`/devices/say?${auth}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (r.ok) setText("");
    else setToast(`say failed (${r.status})`);
  };

  const s = state?.society;
  const voice = state?.voice;
  const label = recording
    ? "Listening… release to send"
    : voice?.status === "transcribing"
      ? "Thinking…"
      : "Hold to talk";

  return (
    <div className="screen-page">
      <div className="device" style={{ position: "relative" }}>
        {!state?.run || !s ? (
          <div className="pad muted">{conn === "live" ? "Waiting for a world with societies…" : conn}</div>
        ) : (
          <>
            <div className="device-head" style={{ background: s.color }}>
              {s.name}
              <span className="n">
                {s.members}
                {s.joined ? ` ▲${s.joined}` : ""}
                {s.left ? ` ▼${s.left}` : ""}
              </span>
            </div>
            <div className="device-body">
              <div>
                <canvas ref={canvas} width={THUMB_W} height={THUMB_H} />
                <div className="device-others">
                  {state.societies
                    ?.filter((o) => o.id !== s.id)
                    .map((o) => (
                      <span key={o.id} style={{ color: o.color }}>
                        ● {o.name} {o.members}
                      </span>
                    ))}
                </div>
              </div>
              <div className="device-threads">
                {state.threads?.length ? (
                  state.threads.map((t, i) => (
                    <div key={i} title={t.title}>
                      {t.title} <span style={{ color: "#888" }}>· {t.replies}</span>
                    </div>
                  ))
                ) : (
                  <div style={{ color: "#888" }}>No threads yet</div>
                )}
              </div>
            </div>
            <div className="device-murmur">
              {voice?.status === "error" ? `⚠ ${voice.error}` : state.murmur ? `“${state.murmur}”` : "Say something to this society."}
            </div>
            <button
              className={`device-talk ${recording ? "rec" : ""}`}
              style={{ background: s.color }}
              onPointerDown={startTalk}
              onPointerUp={stopTalk}
              onPointerLeave={() => recording && stopTalk()}
            >
              {label}
            </button>
            {toast && <div className="device-toast" style={{ top: 34, right: 8 }}>{toast}</div>}
          </>
        )}
      </div>
      <form
        className="screen-say"
        onSubmit={(e) => {
          e.preventDefault();
          say().catch(() => undefined);
        }}
      >
        <input placeholder="…or type what you'd say" value={text} onChange={(e) => setText(e.target.value)} />
        <button type="submit">Say</button>
      </form>
      <p className="muted small">
        Emulator of society screen {society + 1} · {conn}
        {voice?.transcript ? ` · heard: “${voice.transcript}”` : ""}
      </p>
    </div>
  );
}

import type { AgentStatus, NarratorStats, NarratorUpdate, Post, RunState, RunStats, Snapshot, Thought } from "./api";

export const FRAME_SNAPSHOT = 1;
export const FRAME_PIXELS = 2;
const PIXEL_BYTES = 7;

export interface PixelWrite {
  x: number;
  y: number;
  color: number;
  agentId: number;
}

export type ServerEvent =
  | Snapshot
  | { type: "run.state"; state: RunState; stats: RunStats }
  | { type: "agent.status"; agent_id: number; status: AgentStatus }
  | ({ type: "agent.thought"; agent_id: number } & Thought)
  | { type: "post.created"; post: Post; author: string }
  | ({ type: "narrator.update"; stats?: NarratorStats } & NarratorUpdate)
  | { type: "canvas.cleared" }
  | { type: "board.cleared" };

export interface StreamHandlers {
  onEvent: (event: ServerEvent) => void;
  onCanvas: (width: number, height: number, pixels: Uint8Array, owners: Uint16Array) => void;
  onPixels: (writes: PixelWrite[]) => void;
  onClose: (reason: string) => void;
}

export function parsePixelFrame(buf: ArrayBuffer): PixelWrite[] {
  const view = new DataView(buf);
  const out: PixelWrite[] = [];
  for (let off = 1; off + PIXEL_BYTES <= buf.byteLength; off += PIXEL_BYTES) {
    out.push({
      x: view.getUint16(off, true),
      y: view.getUint16(off + 2, true),
      color: view.getUint8(off + 4),
      agentId: view.getUint16(off + 5, true),
    });
  }
  return out;
}

const RECONNECT_MIN_MS = 1000;
const RECONNECT_MAX_MS = 10000;

/** Stream a run's events; reconnects with backoff (each reconnect brings a fresh snapshot). */
export function openRunStream(runId: string, handlers: StreamHandlers): () => void {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  let ws: WebSocket | null = null;
  let retry: number | null = null;
  let delay = RECONNECT_MIN_MS;
  let stopped = false;

  const connect = () => {
    ws = new WebSocket(`${proto}//${location.host}/runs/${runId}/ws`);
    ws.binaryType = "arraybuffer";
    ws.onmessage = (msg: MessageEvent<string | ArrayBuffer>) => {
      if (typeof msg.data === "string") {
        const ev = JSON.parse(msg.data) as ServerEvent;
        if (ev.type === "snapshot") delay = RECONNECT_MIN_MS;
        handlers.onEvent(ev);
        return;
      }
      const buf = msg.data;
      const kind = new Uint8Array(buf, 0, 1)[0];
      if (kind === FRAME_SNAPSHOT) {
        const view = new DataView(buf);
        const width = view.getUint16(1, true);
        const height = view.getUint16(3, true);
        const n = width * height;
        const pixels = new Uint8Array(buf, 5, n);
        // Owners are little-endian u16; copy them through a DataView so alignment and host
        // byte order don't matter.
        const owners = new Uint16Array(n);
        for (let i = 0, off = 5 + n; i < n; i++, off += 2) owners[i] = view.getUint16(off, true);
        handlers.onCanvas(width, height, pixels, owners);
      } else if (kind === FRAME_PIXELS) {
        handlers.onPixels(parsePixelFrame(buf));
      }
    };
    ws.onclose = (e) => {
      ws = null;
      if (stopped) return;
      const reason = e.reason || `closed (${e.code})`;
      // Close codes are unreliable (a refused handshake reaches the browser as 1006, proxies
      // send 1001), so ask the REST API whether the run still exists before retrying.
      void fetch(`/runs/${runId}`)
        .then((r) => r.status !== 404)
        .catch(() => true)
        .then((exists) => {
          if (stopped) return;
          if (!exists) {
            handlers.onClose(`${reason} · run gone`);
            return;
          }
          handlers.onClose(`${reason} · reconnecting in ${Math.round(delay / 1000)} s`);
          retry = window.setTimeout(connect, delay);
          delay = Math.min(delay * 2, RECONNECT_MAX_MS);
        });
    };
  };
  connect();

  return () => {
    stopped = true;
    if (retry !== null) window.clearTimeout(retry);
    ws?.close();
  };
}

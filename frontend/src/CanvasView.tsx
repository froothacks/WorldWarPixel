import { useEffect, useRef, useState, type MouseEvent, type WheelEvent } from "react";
import { GOD_ID, GOD_NAME, type Agent } from "./api";
import type { GodTool } from "./GodBar";
import type { PixelStore } from "./pixelStore";

interface Props {
  store: PixelStore | null;
  version: number;
  agents: Map<number, Agent>;
  highlightAgent: number | null;
  onPickAgent: (agentId: number) => void;
  focus: Region | null;
  /** God-mode brush; when set, dragging paints/erases instead of panning. */
  tool?: GodTool | null;
  onPaint?: (pixels: [number, number, number][], erase: boolean) => void;
}

export interface Region {
  x: number;
  y: number;
  w: number;
  h: number;
  /** Changes on every request so the same region can be focused twice. */
  nonce: number;
}

interface Viewport {
  zoom: number;
  panX: number;
  panY: number;
}

export default function CanvasView({ store, version, agents, highlightAgent, onPickAgent, focus, tool = null, onPaint }: Props) {
  const stroke = useRef<{ cells: Map<number, [number, number, number]>; sent: Set<number> } | null>(null);
  const flushTimer = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  const [vp, setVp] = useState<Viewport | null>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });

  // Track the element's CSS size so the backing store and viewport follow window resizes.
  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    setSize({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null);
  const drag = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);

  // Fit the canvas into the viewport the first time we get a store.
  useEffect(() => {
    if (!store || vp || size.w === 0 || size.h === 0) return;
    const zoom = Math.max(0.1, Math.min(size.w / store.width, size.h / store.height));
    setVp({ zoom, panX: (size.w - store.width * zoom) / 2, panY: (size.h - store.height * zoom) / 2 });
  }, [store, vp, size]);

  // Pan and zoom so a region the narrator pointed at fills about half the viewport.
  useEffect(() => {
    if (!focus || !store || size.w === 0 || size.h === 0) return;
    const zoom = Math.min(64, Math.max(0.1, Math.min(size.w / focus.w, size.h / focus.h) * 0.5));
    const cx = focus.x + focus.w / 2;
    const cy = focus.y + focus.h / 2;
    setVp({ zoom, panX: size.w / 2 - cx * zoom, panY: size.h / 2 - cy * zoom });
    // Only re-run for a new focus request, not when the viewport resizes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focus?.nonce, store]);

  useEffect(() => {
    const el = canvasRef.current;
    if (!el || !store || !vp) return;
    const dpr = window.devicePixelRatio || 1;
    if (el.width !== el.clientWidth * dpr || el.height !== el.clientHeight * dpr) {
      el.width = el.clientWidth * dpr;
      el.height = el.clientHeight * dpr;
    }
    let off = offscreenRef.current;
    if (!off || off.width !== store.width || off.height !== store.height) {
      off = document.createElement("canvas");
      off.width = store.width;
      off.height = store.height;
      offscreenRef.current = off;
    }
    off.getContext("2d")!.putImageData(store.image, 0, 0);
    const ctx = el.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#101014";
    ctx.fillRect(0, 0, el.clientWidth, el.clientHeight);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, vp.panX, vp.panY, store.width * vp.zoom, store.height * vp.zoom);
    if (highlightAgent !== null) {
      ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
      const { owners, width } = store;
      for (let i = 0; i < owners.length; i++) {
        if (owners[i] !== highlightAgent) continue;
        const x = i % width;
        const y = (i - x) / width;
        ctx.fillRect(vp.panX + x * vp.zoom, vp.panY + y * vp.zoom, vp.zoom, vp.zoom);
      }
    }
    if (tool && hover) {
      const half = Math.floor(tool.size / 2);
      ctx.strokeStyle = tool.mode === "erase" ? "#ff5f5f" : "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(vp.panX + (hover.x - half) * vp.zoom, vp.panY + (hover.y - half) * vp.zoom, tool.size * vp.zoom, tool.size * vp.zoom);
    }
    if (focus) {
      ctx.strokeStyle = "#4f8cff";
      ctx.lineWidth = 2;
      ctx.strokeRect(vp.panX + focus.x * vp.zoom, vp.panY + focus.y * vp.zoom, focus.w * vp.zoom, focus.h * vp.zoom);
    }
  }, [store, version, vp, highlightAgent, size, focus, tool, hover]);

  const toPixel = (e: MouseEvent<HTMLCanvasElement>) => {
    if (!store || !vp) return null;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left - vp.panX) / vp.zoom);
    const y = Math.floor((e.clientY - rect.top - vp.panY) / vp.zoom);
    return x >= 0 && y >= 0 && x < store.width && y < store.height ? { x, y } : null;
  };

  const onWheel = (e: WheelEvent<HTMLCanvasElement>) => {
    if (!vp) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.2 : 1 / 1.2;
    const zoom = Math.min(64, Math.max(0.1, vp.zoom * factor));
    const k = zoom / vp.zoom;
    setVp({ zoom, panX: mx - (mx - vp.panX) * k, panY: my - (my - vp.panY) * k });
  };

  // God-mode brush: stamp a size x size square at the pointer, batch and send every 80 ms.
  const stamp = (p: { x: number; y: number } | null) => {
    if (!p || !tool || !store || !stroke.current) return;
    const half = Math.floor(tool.size / 2);
    const color = tool.mode === "erase" ? 0 : tool.color;
    for (let dy = -half; dy < tool.size - half; dy++) {
      for (let dx = -half; dx < tool.size - half; dx++) {
        const x = p.x + dx;
        const y = p.y + dy;
        if (x < 0 || y < 0 || x >= store.width || y >= store.height) continue;
        const key = y * store.width + x;
        if (!stroke.current.sent.has(key)) stroke.current.cells.set(key, [x, y, color]);
      }
    }
    if (flushTimer.current === null) flushTimer.current = window.setTimeout(flushStroke, 80);
  };
  const flushStroke = () => {
    flushTimer.current = null;
    const s = stroke.current;
    if (!s || !tool || s.cells.size === 0) return;
    const batch = [...s.cells.values()];
    s.cells.forEach((_, k) => s.sent.add(k));
    s.cells.clear();
    onPaint?.(batch, tool.mode === "erase");
  };

  const onMouseDown = (e: MouseEvent<HTMLCanvasElement>) => {
    if (tool && e.button === 0) {
      stroke.current = { cells: new Map(), sent: new Set() };
      stamp(toPixel(e));
      return;
    }
    if (vp) drag.current = { x: e.clientX, y: e.clientY, panX: vp.panX, panY: vp.panY };
  };
  const onMouseMove = (e: MouseEvent<HTMLCanvasElement>) => {
    if (stroke.current) stamp(toPixel(e));
    if (drag.current && vp) {
      const d = drag.current;
      setVp({ ...vp, panX: d.panX + e.clientX - d.x, panY: d.panY + e.clientY - d.y });
    }
    setHover(toPixel(e));
  };
  const onMouseUp = (e: MouseEvent<HTMLCanvasElement>) => {
    if (stroke.current) {
      stamp(toPixel(e));
      if (flushTimer.current !== null) window.clearTimeout(flushTimer.current);
      flushStroke();
      stroke.current = null;
      return;
    }
    const d = drag.current;
    drag.current = null;
    if (d && Math.abs(e.clientX - d.x) < 3 && Math.abs(e.clientY - d.y) < 3) {
      const p = toPixel(e);
      const owner = p && store ? store.ownerAt(p.x, p.y) : 0;
      if (owner) onPickAgent(owner);
    }
  };

  const owner = hover && store ? store.ownerAt(hover.x, hover.y) : 0;
  const ownerAgent = owner ? agents.get(owner) : undefined;

  return (
    <div className="canvas-wrap">
      <canvas
        ref={canvasRef}
        className={`canvas ${tool ? "canvas-brush" : ""}`}
        onWheel={onWheel}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={() => {
          drag.current = null;
          if (stroke.current) {
            flushStroke();
            stroke.current = null;
          }
          setHover(null);
        }}
      />
      <div className="canvas-status">
        {store ? `${store.width} × ${store.height}` : "waiting for snapshot…"}
        {vp && ` · zoom ${vp.zoom.toFixed(2)}×`}
        {hover && ` · (${hover.x}, ${hover.y})`}
        {ownerAgent && ` · ${ownerAgent.name} (${ownerAgent.country})`}
        {hover && !ownerAgent && owner === GOD_ID && ` · ${GOD_NAME}`}
        {hover && !ownerAgent && owner !== GOD_ID && " · unclaimed"}
      </div>
    </div>
  );
}

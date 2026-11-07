import { useEffect, useRef, useState, type MouseEvent, type WheelEvent } from "react";
import type { Agent } from "./api";
import type { PixelStore } from "./pixelStore";

interface Props {
  store: PixelStore | null;
  version: number;
  agents: Map<number, Agent>;
  highlightAgent: number | null;
  onPickAgent: (agentId: number) => void;
}

interface Viewport {
  zoom: number;
  panX: number;
  panY: number;
}

export default function CanvasView({ store, version, agents, highlightAgent, onPickAgent }: Props) {
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
  }, [store, version, vp, highlightAgent, size]);

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

  const onMouseDown = (e: MouseEvent<HTMLCanvasElement>) => {
    if (vp) drag.current = { x: e.clientX, y: e.clientY, panX: vp.panX, panY: vp.panY };
  };
  const onMouseMove = (e: MouseEvent<HTMLCanvasElement>) => {
    if (drag.current && vp) {
      const d = drag.current;
      setVp({ ...vp, panX: d.panX + e.clientX - d.x, panY: d.panY + e.clientY - d.y });
    }
    setHover(toPixel(e));
  };
  const onMouseUp = (e: MouseEvent<HTMLCanvasElement>) => {
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
        className="canvas"
        onWheel={onWheel}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={() => {
          drag.current = null;
          setHover(null);
        }}
      />
      <div className="canvas-status">
        {store ? `${store.width} × ${store.height}` : "waiting for snapshot…"}
        {vp && ` · zoom ${vp.zoom.toFixed(2)}×`}
        {hover && ` · (${hover.x}, ${hover.y})`}
        {ownerAgent && ` · ${ownerAgent.name} (${ownerAgent.country})`}
        {hover && !ownerAgent && " · unclaimed"}
      </div>
    </div>
  );
}

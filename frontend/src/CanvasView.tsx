import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { GOD_ID, GOD_NAME, type Agent } from "./api";
import type { GodTool } from "./GodBar";
import type { PixelStore } from "./pixelStore";

interface Props {
  store: PixelStore | null;
  version: number;
  agents: Map<number, Agent>;
  highlightAgent: number | null;
  /** Indexed by owner id; non-zero entries are lit up like highlightAgent (e.g. one country). */
  highlightMask?: Uint8Array | null;
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

interface PointerPosition {
  x: number;
  y: number;
}

interface PinchGesture {
  pointerIds: [number, number];
  startViewport: Viewport;
  startDistance: number;
  startMidpoint: PointerPosition;
}

interface HighlightCache {
  store: PixelStore;
  version: number;
  agent: number | null;
  mask: Uint8Array | null;
  canvas: HTMLCanvasElement;
}

interface SafariGestureEvent extends Event {
  scale?: number;
  clientX?: number;
  clientY?: number;
}

const MIN_ZOOM = 0.1;
const MAX_ZOOM = 64;
const MIN_VISIBLE = 48;

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

export default function CanvasView({
  store,
  version,
  agents,
  highlightAgent,
  highlightMask = null,
  onPickAgent,
  focus,
  tool = null,
  onPaint,
}: Props) {
  const stroke = useRef<{ cells: Map<number, [number, number, number]>; sent: Set<number> } | null>(null);
  const flushTimer = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  const drawnVersionRef = useRef<{ store: PixelStore; version: number } | null>(null);
  const highlightRef = useRef<HighlightCache | null>(null);
  const [vp, setVp] = useState<Viewport | null>(null);
  const vpRef = useRef<Viewport | null>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const fittedStoreRef = useRef<PixelStore | null>(null);
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null);
  const [dragging, setDragging] = useState(false);
  const activePointersRef = useRef(new Map<number, PointerPosition>());
  const dragRef = useRef<{
    pointerId: number;
    x: number;
    y: number;
    panX: number;
    panY: number;
  } | null>(null);
  const pinchRef = useRef<PinchGesture | null>(null);
  const pinchOccurredRef = useRef(false);
  const safariGestureActiveRef = useRef(false);
  const safariGestureScaleRef = useRef(1);

  const applyVp = useCallback(
    (next: Viewport) => {
      if (!store || size.w <= 0 || size.h <= 0) return;
      const zoom = clamp(next.zoom, MIN_ZOOM, MAX_ZOOM);
      const clampPan = (pan: number, canvasSize: number, viewSize: number) => {
        if (canvasSize <= viewSize) return clamp(pan, 0, viewSize - canvasSize);
        const visible = Math.min(MIN_VISIBLE, viewSize);
        return clamp(pan, visible - canvasSize, viewSize - visible);
      };
      const clamped = {
        zoom,
        panX: clampPan(next.panX, store.width * zoom, size.w),
        panY: clampPan(next.panY, store.height * zoom, size.h),
      };
      vpRef.current = clamped;
      setVp(clamped);
    },
    [size, store],
  );

  const fitView = useCallback(() => {
    if (!store || size.w <= 0 || size.h <= 0) return;
    const zoom = Math.max(MIN_ZOOM, Math.min(size.w / store.width, size.h / store.height));
    applyVp({
      zoom,
      panX: (size.w - store.width * zoom) / 2,
      panY: (size.h - store.height * zoom) / 2,
    });
  }, [applyVp, size, store]);

  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const updateSize = () => {
      const w = el.clientWidth;
      const h = el.clientHeight;
      setSize((current) => (current.w === w && current.h === h ? current : { w, h }));
    };
    const ro = new ResizeObserver(updateSize);
    ro.observe(el);
    updateSize();
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!store || size.w === 0 || size.h === 0) return;
    if (fittedStoreRef.current !== store || !vpRef.current) {
      fittedStoreRef.current = store;
      fitView();
      return;
    }
    applyVp(vpRef.current);
  }, [applyVp, fitView, size, store]);

  useEffect(() => {
    if (!focus || !store || size.w === 0 || size.h === 0) return;
    const zoom = clamp(Math.min(size.w / focus.w, size.h / focus.h) * 0.5, MIN_ZOOM, MAX_ZOOM);
    const cx = focus.x + focus.w / 2;
    const cy = focus.y + focus.h / 2;
    applyVp({ zoom, panX: size.w / 2 - cx * zoom, panY: size.h / 2 - cy * zoom });
    // Only re-run for a new focus request, not when the viewport resizes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focus?.nonce, store]);

  useEffect(() => {
    const el = canvasRef.current;
    if (!el || !store || !vp) return;
    const dpr = window.devicePixelRatio || 1;
    const backingWidth = Math.round(el.clientWidth * dpr);
    const backingHeight = Math.round(el.clientHeight * dpr);
    if (el.width !== backingWidth || el.height !== backingHeight) {
      el.width = backingWidth;
      el.height = backingHeight;
    }

    let off = offscreenRef.current;
    if (!off || off.width !== store.width || off.height !== store.height) {
      off = document.createElement("canvas");
      off.width = store.width;
      off.height = store.height;
      offscreenRef.current = off;
      drawnVersionRef.current = null;
    }
    const drawnVersion = drawnVersionRef.current;
    if (drawnVersion?.store !== store || drawnVersion.version !== version) {
      off.getContext("2d")!.putImageData(store.image, 0, 0);
      drawnVersionRef.current = { store, version };
    }

    let highlightCanvas: HTMLCanvasElement | null = null;
    if (highlightAgent !== null || highlightMask) {
      let cached = highlightRef.current;
      if (
        !cached ||
        cached.store !== store ||
        cached.version !== version ||
        cached.agent !== highlightAgent ||
        cached.mask !== highlightMask
      ) {
        const overlay = document.createElement("canvas");
        overlay.width = store.width;
        overlay.height = store.height;
        const overlayCtx = overlay.getContext("2d")!;
        const image = overlayCtx.createImageData(store.width, store.height);
        for (let i = 0; i < store.owners.length; i++) {
          const owner = store.owners[i];
          if (owner !== highlightAgent && !(highlightMask && highlightMask[owner])) continue;
          const offset = i * 4;
          image.data[offset] = 255;
          image.data[offset + 1] = 255;
          image.data[offset + 2] = 255;
          image.data[offset + 3] = 217;
        }
        overlayCtx.putImageData(image, 0, 0);
        cached = { store, version, agent: highlightAgent, mask: highlightMask, canvas: overlay };
        highlightRef.current = cached;
      }
      highlightCanvas = cached.canvas;
    } else {
      highlightRef.current = null;
    }

    const ctx = el.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#0b0b0c";
    ctx.fillRect(0, 0, el.clientWidth, el.clientHeight);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, vp.panX, vp.panY, store.width * vp.zoom, store.height * vp.zoom);
    if (highlightCanvas)
      ctx.drawImage(
        highlightCanvas,
        vp.panX,
        vp.panY,
        store.width * vp.zoom,
        store.height * vp.zoom,
      );
    if (tool && hover) {
      const half = Math.floor(tool.size / 2);
      ctx.strokeStyle = tool.mode === "erase" ? "#ff5f5f" : "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(
        vp.panX + (hover.x - half) * vp.zoom,
        vp.panY + (hover.y - half) * vp.zoom,
        tool.size * vp.zoom,
        tool.size * vp.zoom,
      );
    }
    if (focus) {
      ctx.strokeStyle = "#ff6b3d";
      ctx.lineWidth = 2;
      ctx.strokeRect(
        vp.panX + focus.x * vp.zoom,
        vp.panY + focus.y * vp.zoom,
        focus.w * vp.zoom,
        focus.h * vp.zoom,
      );
    }
  }, [store, version, vp, highlightAgent, highlightMask, size, focus, tool, hover]);

  const toPixel = (clientX: number, clientY: number) => {
    const current = vpRef.current;
    const el = canvasRef.current;
    if (!store || !current || !el) return null;
    const rect = el.getBoundingClientRect();
    const x = Math.floor((clientX - rect.left - current.panX) / current.zoom);
    const y = Math.floor((clientY - rect.top - current.panY) / current.zoom);
    return x >= 0 && y >= 0 && x < store.width && y < store.height ? { x, y } : null;
  };

  const flushStroke = () => {
    flushTimer.current = null;
    const currentStroke = stroke.current;
    if (!currentStroke || !tool || currentStroke.cells.size === 0) return;
    const batch = [...currentStroke.cells.values()];
    currentStroke.cells.forEach((_, key) => currentStroke.sent.add(key));
    currentStroke.cells.clear();
    onPaint?.(batch, tool.mode === "erase");
  };

  const stamp = (point: { x: number; y: number } | null) => {
    if (!point || !tool || !store || !stroke.current) return;
    const half = Math.floor(tool.size / 2);
    const color = tool.mode === "erase" ? 0 : tool.color;
    for (let dy = -half; dy < tool.size - half; dy++) {
      for (let dx = -half; dx < tool.size - half; dx++) {
        const x = point.x + dx;
        const y = point.y + dy;
        if (x < 0 || y < 0 || x >= store.width || y >= store.height) continue;
        const key = y * store.width + x;
        if (!stroke.current.sent.has(key)) stroke.current.cells.set(key, [x, y, color]);
      }
    }
    if (flushTimer.current === null) flushTimer.current = window.setTimeout(flushStroke, 80);
  };

  const finishStroke = (point: { x: number; y: number } | null = null) => {
    if (point) stamp(point);
    if (flushTimer.current !== null) {
      window.clearTimeout(flushTimer.current);
      flushTimer.current = null;
    }
    flushStroke();
    stroke.current = null;
  };

  const startDrag = (pointerId: number, position: PointerPosition) => {
    const current = vpRef.current;
    if (!current) return;
    dragRef.current = {
      pointerId,
      x: position.x,
      y: position.y,
      panX: current.panX,
      panY: current.panY,
    };
    setDragging(true);
  };

  const startPinch = () => {
    const pointers = [...activePointersRef.current.entries()].slice(0, 2);
    const current = vpRef.current;
    const el = canvasRef.current;
    if (pointers.length < 2 || !current || !el) return;
    const [first, second] = pointers;
    const rect = el.getBoundingClientRect();
    const dx = first[1].x - second[1].x;
    const dy = first[1].y - second[1].y;
    pinchRef.current = {
      pointerIds: [first[0], second[0]],
      startViewport: current,
      startDistance: Math.max(1, Math.hypot(dx, dy)),
      startMidpoint: {
        x: (first[1].x + second[1].x) / 2 - rect.left,
        y: (first[1].y + second[1].y) / 2 - rect.top,
      },
    };
  };

  const updatePinch = () => {
    const gesture = pinchRef.current;
    const el = canvasRef.current;
    if (!gesture || !el) return;
    const first = activePointersRef.current.get(gesture.pointerIds[0]);
    const second = activePointersRef.current.get(gesture.pointerIds[1]);
    if (!first || !second) return;
    const rect = el.getBoundingClientRect();
    const distance = Math.hypot(first.x - second.x, first.y - second.y);
    const midpoint = {
      x: (first.x + second.x) / 2 - rect.left,
      y: (first.y + second.y) / 2 - rect.top,
    };
    const zoom = clamp(
      gesture.startViewport.zoom * (distance / gesture.startDistance),
      MIN_ZOOM,
      MAX_ZOOM,
    );
    const scale = zoom / gesture.startViewport.zoom;
    applyVp({
      zoom,
      panX: midpoint.x - (gesture.startMidpoint.x - gesture.startViewport.panX) * scale,
      panY: midpoint.y - (gesture.startMidpoint.y - gesture.startViewport.panY) * scale,
    });
  };

  const onPointerDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch {
      // The pointer may already have been released by the browser.
    }
    activePointersRef.current.set(event.pointerId, { x: event.clientX, y: event.clientY });

    if (activePointersRef.current.size >= 2) {
      if (stroke.current) finishStroke();
      dragRef.current = null;
      pinchOccurredRef.current = true;
      if (!pinchRef.current) startPinch();
      setDragging(true);
      setHover(null);
      return;
    }

    if (tool && event.button === 0) {
      stroke.current = { cells: new Map(), sent: new Set() };
      dragRef.current = null;
      setDragging(false);
      stamp(toPixel(event.clientX, event.clientY));
      return;
    }
    startDrag(event.pointerId, { x: event.clientX, y: event.clientY });
    if (event.pointerType !== "mouse") setHover(null);
  };

  const onPointerMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const active = activePointersRef.current.has(event.pointerId);
    if (active) activePointersRef.current.set(event.pointerId, { x: event.clientX, y: event.clientY });

    if (active && activePointersRef.current.size >= 2) {
      updatePinch();
      return;
    }
    if (active && stroke.current) {
      stamp(toPixel(event.clientX, event.clientY));
    } else if (active && dragRef.current?.pointerId === event.pointerId) {
      const drag = dragRef.current;
      const current = vpRef.current;
      if (current) {
        applyVp({
          ...current,
          panX: drag.panX + event.clientX - drag.x,
          panY: drag.panY + event.clientY - drag.y,
        });
      }
    }
    if (event.pointerType === "mouse") setHover(toPixel(event.clientX, event.clientY));
  };

  const onPointerUp = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!activePointersRef.current.has(event.pointerId)) return;
    activePointersRef.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    const wasBrushing = stroke.current !== null;
    if (wasBrushing) finishStroke(toPixel(event.clientX, event.clientY));
    const drag = dragRef.current?.pointerId === event.pointerId ? dragRef.current : null;
    activePointersRef.current.delete(event.pointerId);

    if (activePointersRef.current.size >= 2) {
      pinchRef.current = null;
      startPinch();
      pinchOccurredRef.current = true;
      dragRef.current = null;
      return;
    }
    if (activePointersRef.current.size === 1) {
      pinchRef.current = null;
      const [pointerId, position] = [...activePointersRef.current.entries()][0];
      startDrag(pointerId, position);
      return;
    }

    pinchRef.current = null;
    dragRef.current = null;
    setDragging(false);
    if (
      !wasBrushing &&
      !pinchOccurredRef.current &&
      !tool &&
      drag &&
      Math.hypot(event.clientX - drag.x, event.clientY - drag.y) < 3
    ) {
      const point = toPixel(event.clientX, event.clientY);
      const owner = point && store ? store.ownerAt(point.x, point.y) : 0;
      if (owner) onPickAgent(owner);
    }
    pinchOccurredRef.current = false;
  };

  const onPointerCancel = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!activePointersRef.current.has(event.pointerId)) return;
    if (stroke.current) finishStroke();
    activePointersRef.current.delete(event.pointerId);
    if (activePointersRef.current.size >= 2) {
      pinchRef.current = null;
      startPinch();
      pinchOccurredRef.current = true;
    } else if (activePointersRef.current.size === 1) {
      pinchRef.current = null;
      const [pointerId, position] = [...activePointersRef.current.entries()][0];
      startDrag(pointerId, position);
    } else {
      pinchRef.current = null;
      dragRef.current = null;
      pinchOccurredRef.current = false;
      setDragging(false);
    }
  };

  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;

    const zoomAt = (current: Viewport, x: number, y: number, rawFactor: number) => {
      const zoom = clamp(current.zoom * clamp(rawFactor, 0.5, 2), MIN_ZOOM, MAX_ZOOM);
      const scale = zoom / current.zoom;
      return {
        zoom,
        panX: x - (x - current.panX) * scale,
        panY: y - (y - current.panY) * scale,
      };
    };

    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      if ((event.ctrlKey || event.metaKey) && safariGestureActiveRef.current) return;
      const current = vpRef.current;
      if (!store || !current) return;

      const multiplier =
        event.deltaMode === WheelEvent.DOM_DELTA_LINE
          ? 16
          : event.deltaMode === WheelEvent.DOM_DELTA_PAGE
            ? window.innerHeight
            : 1;
      const dx = event.deltaX * multiplier;
      const dy = event.deltaY * multiplier;
      const rect = el.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      if (event.ctrlKey || event.metaKey) {
        applyVp(zoomAt(current, x, y, Math.exp(-dy * 0.01)));
        return;
      }

      let next = current;
      if (!event.shiftKey && dy !== 0)
        next = zoomAt(current, x, y, Math.exp(-dy * 0.0015));
      const horizontalDelta = event.shiftKey ? dx || dy : dx;
      if (horizontalDelta !== 0) next = { ...next, panX: next.panX - horizontalDelta };
      applyVp(next);
    };

    const onGestureStart = (event: Event) => {
      event.preventDefault();
      safariGestureActiveRef.current = true;
      const gesture = event as SafariGestureEvent;
      safariGestureScaleRef.current = gesture.scale ?? 1;
    };

    const onGestureChange = (event: Event) => {
      event.preventDefault();
      const gesture = event as SafariGestureEvent;
      const scale = gesture.scale ?? safariGestureScaleRef.current;
      const previousScale = safariGestureScaleRef.current || 1;
      safariGestureScaleRef.current = scale;
      const current = vpRef.current;
      if (!current) return;
      const rect = el.getBoundingClientRect();
      const clientX = gesture.clientX ?? rect.left + rect.width / 2;
      const clientY = gesture.clientY ?? rect.top + rect.height / 2;
      const factor = scale / previousScale;
      const zoom = clamp(current.zoom * clamp(factor, 0.5, 2), MIN_ZOOM, MAX_ZOOM);
      const ratio = zoom / current.zoom;
      applyVp({
        zoom,
        panX: clientX - rect.left - (clientX - rect.left - current.panX) * ratio,
        panY: clientY - rect.top - (clientY - rect.top - current.panY) * ratio,
      });
    };

    const onGestureEnd = (event: Event) => {
      event.preventDefault();
      safariGestureActiveRef.current = false;
      safariGestureScaleRef.current = 1;
    };

    el.addEventListener("wheel", onWheel, { passive: false });
    el.addEventListener("gesturestart", onGestureStart, { passive: false });
    el.addEventListener("gesturechange", onGestureChange, { passive: false });
    el.addEventListener("gestureend", onGestureEnd, { passive: false });
    return () => {
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("gesturestart", onGestureStart);
      el.removeEventListener("gesturechange", onGestureChange);
      el.removeEventListener("gestureend", onGestureEnd);
    };
  }, [applyVp, store]);

  useEffect(
    () => () => {
      if (flushTimer.current !== null) window.clearTimeout(flushTimer.current);
    },
    [],
  );

  const owner = hover && store ? store.ownerAt(hover.x, hover.y) : 0;
  const ownerAgent = owner ? agents.get(owner) : undefined;

  return (
    <div className="canvas-wrap">
      <canvas
        ref={canvasRef}
        className={`canvas ${tool ? "canvas-brush" : ""}${dragging ? " dragging" : ""}`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerCancel}
        onLostPointerCapture={onPointerCancel}
        onPointerLeave={() => setHover(null)}
      />
      <button
        type="button"
        className="canvas-fit ghost"
        aria-label="Fit canvas to view"
        disabled={!store}
        onClick={fitView}
      >
        fit
      </button>
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

import { useEffect, useRef } from "react";

const COLORS = ["#E50000", "#E59500", "#E5D900", "#94E044", "#02BE01", "#00D3DD", "#0083C7", "#0000EA", "#CF6EE4", "#820080", "#FFA7D1", "#A06A42"];
const CELL = 8;

interface Painter {
  x: number;
  y: number;
  hx: number;
  hy: number;
  r: number;
  color: string;
}

/** Decorative background: painters growing small territories on the grid, slowly fading. */
export default function PixelField() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let cols = 0;
    let rows = 0;
    let painters: Painter[] = [];
    let raf = 0;
    let last = 0;
    let n = 0;

    const rand = (n: number) => Math.floor(Math.random() * n);
    const spawn = (color: string): Painter => {
      const hx = rand(cols);
      const hy = rand(rows);
      return { x: hx, y: hy, hx, hy, r: 3 + rand(6), color };
    };

    const paint = (p: Painter) => {
      p.x += rand(3) - 1;
      p.y += rand(3) - 1;
      if (Math.abs(p.x - p.hx) > p.r) p.x -= Math.sign(p.x - p.hx);
      if (Math.abs(p.y - p.hy) > p.r) p.y -= Math.sign(p.y - p.hy);
      if (Math.random() < 0.02) {
        p.hx = Math.min(cols - 1, Math.max(0, p.hx + rand(5) - 2));
        p.hy = Math.min(rows - 1, Math.max(0, p.hy + rand(5) - 2));
      }
      ctx.fillStyle = p.color;
      ctx.fillRect(p.x * CELL + 1, p.y * CELL + 1, CELL - 1, CELL - 1);
    };

    const fade = () => {
      ctx.globalCompositeOperation = "destination-out";
      ctx.fillStyle = "rgba(0, 0, 0, 0.08)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.globalCompositeOperation = "source-over";
    };

    const step = (i: number) => {
      painters.forEach(paint);
      if (i % 13 === 0) fade();
      if (Math.random() < 0.01) painters[rand(painters.length)] = spawn(COLORS[rand(COLORS.length)]);
    };

    const resize = () => {
      const { width, height } = canvas.getBoundingClientRect();
      canvas.width = Math.ceil(width);
      canvas.height = Math.ceil(height);
      cols = Math.ceil(width / CELL);
      rows = Math.ceil(height / CELL);
      painters = Array.from({ length: 22 }, (_, i) => spawn(COLORS[i % COLORS.length]));
      for (let i = 1; i <= 160; i++) step(i);
    };

    const tick = (t: number) => {
      raf = requestAnimationFrame(tick);
      if (t - last < 60) return;
      last = t;
      step(++n);
    };

    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    if (!reduced) raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, []);

  return <canvas ref={ref} className="pixel-field" aria-hidden />;
}

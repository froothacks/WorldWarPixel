import { useEffect, useMemo, useRef, useState } from "react";
import { api, type Agent, type WorldEdge, type WorldGraph } from "./api";

interface Props {
  runId: string;
  running: boolean;
  agents: Agent[];
  palette: string[];
  highlight: string | null;
  onHighlight: (country: string | null) => void;
  hidden?: boolean;
}

const POLL_MS = 3000;
/** Keep the picture readable: a handful of the most active countries and their strongest ties. */
const MAX_NODES = 10;
const MAX_EDGES = 14;

type Relation = "allies" | "rivals";

const weight = (e: WorldEdge) => e.talk * 10 + e.raids_ab + e.raids_ba;
const relation = (e: WorldEdge): Relation => (e.talk * 10 >= e.raids_ab + e.raids_ba ? "allies" : "rivals");
const STROKE: Record<Relation, string> = {
  allies: "#7aa7ff",
  rivals: "#e5534b",
};

export default function Alliances({ runId, running, agents, palette, highlight, onHighlight, hidden = false }: Props) {
  const [graph, setGraph] = useState<WorldGraph | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [size, setSize] = useState({ w: 300, h: 300 });
  const wrap = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    const poll = () =>
      api
        .world(runId)
        .then((g) => {
          if (cancelled) return;
          setGraph(g);
          setError(null);
        })
        .catch((e: Error) => !cancelled && setError(e.message));
    poll();
    if (!running) return () => void (cancelled = true);
    const t = window.setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(t);
    };
  }, [runId, running]);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    setSize({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  const colorOf = useMemo(() => {
    const tally = new Map<string, Map<number, number>>();
    for (const a of agents) {
      const m = tally.get(a.country) ?? new Map<number, number>();
      m.set(a.color, (m.get(a.color) ?? 0) + 1);
      tally.set(a.country, m);
    }
    const out = new Map<string, string>();
    tally.forEach((m, country) => {
      let best = 0;
      let n = -1;
      m.forEach((k, c) => {
        if (k > n) {
          n = k;
          best = c;
        }
      });
      out.set(country, palette[best] ?? "#888");
    });
    return out;
  }, [agents, palette]);

  const shown = useMemo(() => {
    if (!graph) return { nodes: [] as string[], edges: [] as WorldEdge[] };
    const pop = new Map(graph.nodes.map((n) => [n.country, n.agents]));
    const hit = (e: WorldEdge) => !!highlight && (e.a === highlight || e.b === highlight);
    // highlighted country's ties first, then strongest
    const ranked = [...graph.edges].sort((a, b) => Number(hit(b)) - Number(hit(a)) || weight(b) - weight(a));
    const keep = new Set<string>();
    if (highlight && pop.has(highlight)) keep.add(highlight);
    for (const e of ranked) {
      if (keep.size >= MAX_NODES) break;
      const add = [e.a, e.b].filter((n) => !keep.has(n));
      if (keep.size + add.length > MAX_NODES) continue;
      for (const n of add) keep.add(n);
    }
    // quiet runs: fill the ring with the most populous countries
    for (const n of [...graph.nodes].sort((a, b) => b.agents - a.agents)) {
      if (keep.size >= MAX_NODES) break;
      keep.add(n.country);
    }
    const edges = ranked.filter((e) => keep.has(e.a) && keep.has(e.b)).slice(0, MAX_EDGES);
    // Seat by population so the ring order only changes when the cast does.
    const nodes = [...keep].sort((a, b) => (pop.get(b) ?? 0) - (pop.get(a) ?? 0) || a.localeCompare(b));
    return { nodes, edges };
  }, [graph, highlight]);

  const layout = useMemo(() => {
    const { w, h } = size;
    const cx = w / 2;
    const cy = h / 2;
    // an ellipse uses the tall, narrow panel; labels sit above/below their node and are clamped inside
    const rx = Math.max(30, w / 2 - 30);
    const ry = Math.max(30, h / 2 - 30);
    const pos = new Map<string, { x: number; y: number; angle: number }>();
    shown.nodes.forEach((name, i) => {
      const angle = -Math.PI / 2 + (i / Math.max(1, shown.nodes.length)) * Math.PI * 2;
      pos.set(name, { x: cx + Math.cos(angle) * rx, y: cy + Math.sin(angle) * ry, angle });
    });
    return { pos, cx, cy };
  }, [shown, size]);

  const pop = useMemo(() => new Map(graph?.nodes.map((n) => [n.country, n.agents]) ?? []), [graph]);
  const maxPop = Math.max(1, ...shown.nodes.map((n) => pop.get(n) ?? 1));
  const maxW = Math.max(1, ...shown.edges.map(weight));
  const radius = (country: string) => 5 + 9 * Math.sqrt((pop.get(country) ?? 1) / maxPop);
  const edgeTitle = (e: WorldEdge) =>
    `${e.a} ↔ ${e.b}: ${e.talk} messages · ${e.a} painted over ${e.raids_ab.toLocaleString()} px of ${e.b}, ${e.b} over ${e.raids_ba.toLocaleString()} px of ${e.a}`;
  const rivals = graph
    ? graph.edges
        .filter((e) => relation(e) === "rivals")
        .sort((a, b) => weight(b) - weight(a))
        .slice(0, 3)
    : [];
  const allies = graph
    ? graph.edges
        .filter((e) => e.talk > 0)
        .sort((a, b) => b.talk - a.talk)
        .slice(0, 3)
    : [];
  const short = (name: string) => (name.length > 12 ? `${name.slice(0, 11)}…` : name);
  const touches = (e: WorldEdge) => !highlight || e.a === highlight || e.b === highlight;

  return (
    <aside className="agents alliances" style={hidden ? { display: "none" } : undefined}>
      <header className="panel-head">
        <span className="label">Alliances</span>
        <span className="count">
          <span style={{ color: STROKE.allies }}>talk</span> · <span style={{ color: STROKE.rivals }}>raids</span>
        </span>
        {highlight && (
          <button className="ghost" onClick={() => onHighlight(null)}>
            clear
          </button>
        )}
      </header>
      <div className="graph" ref={wrap}>
        {error && <p className="muted small pad">{error}</p>}
        {graph && graph.edges.length === 0 && <p className="muted small pad empty">No cross-border replies or raids yet.</p>}
        <svg width={size.w} height={size.h}>
          {shown.edges.map((e) => {
            const a = layout.pos.get(e.a);
            const b = layout.pos.get(e.b);
            if (!a || !b) return null;
            const rel = relation(e);
            const on = touches(e);
            // bow each chord toward the centre so crossing lines stay distinguishable
            const mx = (a.x + b.x) / 2;
            const my = (a.y + b.y) / 2;
            const qx = mx + (layout.cx - mx) * 0.45;
            const qy = my + (layout.cy - my) * 0.45;
            return (
              <path
                key={`${e.a}|${e.b}`}
                d={`M${a.x},${a.y} Q${qx},${qy} ${b.x},${b.y}`}
                fill="none"
                stroke={STROKE[rel]}
                strokeOpacity={on ? 0.85 : 0.08}
                strokeWidth={1 + 4 * (weight(e) / maxW)}
                strokeLinecap="round"
              >
                <title>{edgeTitle(e)}</title>
              </path>
            );
          })}
          {shown.nodes.map((name) => {
            const n = layout.pos.get(name);
            if (!n) return null;
            const r = radius(name);
            const on = !highlight || highlight === name || shown.edges.some((e) => touches(e) && (e.a === name || e.b === name));
            const label = short(name);
            const half = label.length * 3 + 2;
            const above = Math.sin(n.angle) < -0.3;
            return (
              <g key={name} className="node" opacity={on ? 1 : 0.3} onClick={() => onHighlight(highlight === name ? null : name)}>
                <circle
                  cx={n.x}
                  cy={n.y}
                  r={r}
                  fill={colorOf.get(name) ?? "#888"}
                  stroke={highlight === name ? "#fff" : "#0b0b0c"}
                  strokeWidth={highlight === name ? 2 : 1}
                />
                <text x={Math.min(size.w - half, Math.max(half, n.x))} y={above ? n.y - r - 4 : n.y + r + 11} textAnchor="middle">
                  {label}
                </text>
                <title>
                  {name} · {pop.get(name) ?? 0} agents
                </title>
              </g>
            );
          })}
        </svg>
      </div>
      {(rivals.length > 0 || allies.length > 0) && (
        <div className="rel-lists">
          {rivals.length > 0 && (
            <div>
              <span className="label" style={{ color: STROKE.rivals }}>
                Wars
              </span>
              {rivals.map((e) => (
                <button key={`${e.a}|${e.b}`} className="rel" onClick={() => onHighlight(e.raids_ab >= e.raids_ba ? e.a : e.b)} title={edgeTitle(e)}>
                  <span className="rel-name">
                    {e.a} vs {e.b}
                  </span>
                  <span className="muted">{(e.raids_ab + e.raids_ba).toLocaleString()} px</span>
                </button>
              ))}
            </div>
          )}
          {allies.length > 0 && (
            <div>
              <span className="label" style={{ color: STROKE.allies }}>
                Talking
              </span>
              {allies.map((e) => (
                <button key={`${e.a}|${e.b}`} className="rel" onClick={() => onHighlight(e.a)} title={edgeTitle(e)}>
                  <span className="rel-name">
                    {e.a} ↔ {e.b}
                  </span>
                  <span className="muted">{e.talk} msgs</span>
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </aside>
  );
}

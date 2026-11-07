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

interface Node {
  country: string;
  agents: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
}

const POLL_MS = 3000;
/** Only the strongest relations are drawn; random brains raid everyone, which is unreadable as a graph. */
const MAX_EDGES = 60;
const MAX_NODES = 40;
const TICKS_PER_UPDATE = 150;

type Relation = "allies" | "rivals" | "tense";

function relation(e: WorldEdge): Relation {
  const raids = e.raids_ab + e.raids_ba;
  if (raids === 0) return "allies";
  if (e.talk === 0) return "rivals";
  const r = raids / (raids + e.talk * 10);
  return r > 0.6 ? "rivals" : r < 0.3 ? "allies" : "tense";
}

const STROKE: Record<Relation, string> = { allies: "#7aa7ff", rivals: "#e5534b", tense: "#d9b44a" };

export default function Alliances({ runId, running, agents, palette, highlight, onHighlight, hidden = false }: Props) {
  const [graph, setGraph] = useState<WorldGraph | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [size, setSize] = useState({ w: 300, h: 300 });
  const wrap = useRef<HTMLDivElement>(null);
  const nodes = useRef(new Map<string, Node>());
  const [pos, setPos] = useState(new Map<string, Node>());

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
    const weight = (e: WorldEdge) => e.talk * 10 + e.raids_ab + e.raids_ba;
    const strongest = [...graph.edges].sort((a, b) => weight(b) - weight(a));
    const floor = strongest.length ? weight(strongest[0]) * 0.05 : 0;
    const edges: WorldEdge[] = [];
    const names: string[] = [];
    const keep = new Set<string>();
    for (const e of strongest) {
      if (edges.length >= MAX_EDGES || weight(e) < floor) break;
      const add = [e.a, e.b].filter((n) => !keep.has(n));
      if (keep.size + add.length > MAX_NODES) continue;
      for (const n of add) {
        keep.add(n);
        names.push(n);
      }
      edges.push(e);
    }
    if (names.length === 0)
      for (const n of [...graph.nodes].sort((a, b) => b.agents - a.agents).slice(0, 12)) names.push(n.country);
    return { nodes: names, edges };
  }, [graph]);

  // Fruchterman-Reingold with a cooling temperature: a short burst of ticks after each poll,
  // positions persist between polls so the picture does not jump.
  useEffect(() => {
    if (!graph) return;
    const { w, h } = size;
    if (w < 40 || h < 40) return;
    const pop = new Map(graph.nodes.map((n) => [n.country, n.agents]));
    const live = nodes.current;
    for (const name of shown.nodes) {
      const n = live.get(name);
      if (n) n.agents = pop.get(name) ?? 1;
      else {
        const a = Math.random() * Math.PI * 2;
        const r = Math.min(w, h) * (0.2 + Math.random() * 0.2);
        live.set(name, { country: name, agents: pop.get(name) ?? 1, x: w / 2 + Math.cos(a) * r, y: h / 2 + Math.sin(a) * r, vx: 0, vy: 0 });
      }
    }
    for (const k of [...live.keys()]) if (!shown.nodes.includes(k)) live.delete(k);
    const list = [...live.values()];
    const maxW = shown.edges.reduce((m, e) => Math.max(m, e.talk * 10 + e.raids_ab + e.raids_ba), 1);
    const k = Math.sqrt((w * h) / Math.max(2, list.length)) * 0.8;
    let t = Math.min(w, h) / 6;
    const cool = Math.pow(1 / t, 1 / TICKS_PER_UPDATE);
    let ticks = 0;
    let raf = 0;
    const step = () => {
      for (const n of list) {
        n.vx = (w / 2 - n.x) * 0.05;
        n.vy = (h / 2 - n.y) * 0.05;
      }
      for (let i = 0; i < list.length; i++)
        for (let j = i + 1; j < list.length; j++) {
          const a = list[i];
          const b = list[j];
          let dx = a.x - b.x;
          let dy = a.y - b.y;
          let d = Math.hypot(dx, dy);
          if (d < 1) {
            dx = Math.random() - 0.5;
            dy = Math.random() - 0.5;
            d = 1;
          }
          const f = (k * k) / d;
          a.vx += (dx / d) * f;
          a.vy += (dy / d) * f;
          b.vx -= (dx / d) * f;
          b.vy -= (dy / d) * f;
        }
      for (const e of shown.edges) {
        const a = live.get(e.a);
        const b = live.get(e.b);
        if (!a || !b) continue;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const d = Math.max(1, Math.hypot(dx, dy));
        const strength = 0.3 + (0.7 * (e.talk * 10 + e.raids_ab + e.raids_ba)) / maxW;
        const f = ((d * d) / k) * strength;
        a.vx += (dx / d) * f;
        a.vy += (dy / d) * f;
        b.vx -= (dx / d) * f;
        b.vy -= (dy / d) * f;
      }
      for (const n of list) {
        const d = Math.hypot(n.vx, n.vy);
        const m = Math.min(d, t) / Math.max(d, 1e-6);
        n.x = Math.min(w - 28, Math.max(28, n.x + n.vx * m));
        n.y = Math.min(h - 28, Math.max(28, n.y + n.vy * m));
      }
      t = Math.max(0.5, t * cool);
      setPos(new Map(list.map((n) => [n.country, { ...n }])));
      if (++ticks < TICKS_PER_UPDATE) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [graph, shown, size]);

  const radius = (agents: number) => 5 + Math.sqrt(agents) * 2.2;
  const edgeTitle = (e: WorldEdge) =>
    `${e.a} ↔ ${e.b}: ${e.talk} replies/mentions · ${e.a} overpainted ${e.raids_ab.toLocaleString()} px of ${e.b}, ${e.b} overpainted ${e.raids_ba.toLocaleString()} px of ${e.a}`;
  const rivals = graph ? graph.edges.filter((e) => relation(e) === "rivals").sort((a, b) => b.raids_ab + b.raids_ba - a.raids_ab - a.raids_ba).slice(0, 3) : [];
  const allies = graph ? graph.edges.filter((e) => relation(e) === "allies").sort((a, b) => b.talk - a.talk).slice(0, 3) : [];
  return (
    <aside className="agents alliances" style={hidden ? { display: "none" } : undefined}>
      <header className="panel-head">
        <span className="label">Alliances</span>
        <span className="count">{graph ? `${graph.edges.length} relations` : "…"}</span>
        {highlight && (
          <button className="ghost" onClick={() => onHighlight(null)}>
            clear
          </button>
        )}
      </header>
      <p className="muted small pad legend">
        <span style={{ color: STROKE.allies }}>— talk</span> (replies, @mentions) <span style={{ color: STROKE.rivals }}>— raids</span> (painting over
        each other) <span style={{ color: STROKE.tense }}>— both</span>. Node size is population. Click a country to light up its land.
      </p>
      <div className="graph" ref={wrap}>
        {error && <p className="muted small pad">{error}</p>}
        {graph && shown.edges.length === 0 && <p className="muted small pad empty">No cross-border replies or raids yet.</p>}
        <svg width={size.w} height={size.h}>
          {shown.edges.map((e) => {
            const a = pos.get(e.a);
            const b = pos.get(e.b);
            if (!a || !b) return null;
            const rel = relation(e);
            const dim = highlight && highlight !== e.a && highlight !== e.b;
            return (
              <line
                key={`${e.a}|${e.b}`}
                x1={a.x}
                y1={a.y}
                x2={b.x}
                y2={b.y}
                stroke={STROKE[rel]}
                strokeOpacity={dim ? 0.12 : 0.7}
                strokeWidth={1 + Math.log2(1 + e.talk + e.raids_ab + e.raids_ba)}
                strokeDasharray={rel === "rivals" ? "4 3" : undefined}
              >
                <title>{edgeTitle(e)}</title>
              </line>
            );
          })}
          {shown.nodes.map((name) => {
            const n = pos.get(name);
            if (!n) return null;
            const r = radius(n.agents);
            const dim = highlight && highlight !== name;
            return (
              <g key={name} className="node" opacity={dim ? 0.35 : 1} onClick={() => onHighlight(highlight === name ? null : name)}>
                <circle cx={n.x} cy={n.y} r={r} fill={colorOf.get(name) ?? "#888"} stroke={highlight === name ? "#fff" : "#0b0b0c"} strokeWidth={highlight === name ? 2 : 1} />
                <text x={n.x} y={n.y + r + 11} textAnchor="middle">
                  {name}
                </text>
                <title>
                  {name} · {n.agents} agents
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
                  {e.raids_ab >= e.raids_ba ? `${e.a} → ${e.b}` : `${e.b} → ${e.a}`}
                  <span className="muted">{(e.raids_ab + e.raids_ba).toLocaleString()} px</span>
                </button>
              ))}
            </div>
          )}
          {allies.length > 0 && (
            <div>
              <span className="label" style={{ color: STROKE.allies }}>
                Allies
              </span>
              {allies.map((e) => (
                <button key={`${e.a}|${e.b}`} className="rel" onClick={() => onHighlight(e.a)} title={edgeTitle(e)}>
                  {e.a} ↔ {e.b}
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

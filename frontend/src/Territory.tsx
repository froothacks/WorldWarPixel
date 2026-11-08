import { useEffect, useMemo, useRef, useState } from "react";
import { GOD_ID, type Agent } from "./api";
import type { PixelStore } from "./pixelStore";

interface Props {
  store: PixelStore | null;
  version: number;
  agents: Agent[];
  palette: string[];
  highlight: string | null;
  onHighlight: (country: string | null) => void;
  /** What a territory unit is called in the copy ("countries", or "people" in roster runs). */
  units?: string;
  hidden?: boolean;
}

interface Row {
  country: string;
  pixels: number;
  delta: number;
  agents: number;
  color: string;
}

/** How often the owners map is re-counted; a 512x320 canvas takes ~1 ms, 4k x 4k ~100 ms. */
const RECOUNT_MS = 1000;
/** Deltas compare against a count taken this long ago. */
const TREND_MS = 15_000;

export default function Territory({ store, version, agents, palette, highlight, onHighlight, hidden = false, units = "countries" }: Props) {
  const countries = useMemo(() => {
    const names: string[] = [];
    const index = new Map<string, number>();
    const perCountry = new Map<number, number>();
    const colors = new Map<number, Map<number, number>>();
    const ownerToCountry = new Uint16Array(65536);
    for (const a of agents) {
      let i = index.get(a.country);
      if (i === undefined) {
        i = names.length;
        names.push(a.country);
        index.set(a.country, i);
      }
      ownerToCountry[a.id] = i + 1;
      perCountry.set(i, (perCountry.get(i) ?? 0) + 1);
      const c = colors.get(i) ?? new Map<number, number>();
      c.set(a.color, (c.get(a.color) ?? 0) + 1);
      colors.set(i, c);
    }
    const color = names.map((_, i) => {
      let best = 0;
      let bestN = -1;
      colors.get(i)?.forEach((n, col) => {
        if (n > bestN) {
          bestN = n;
          best = col;
        }
      });
      return palette[best] ?? "#888";
    });
    return { names, ownerToCountry, agents: names.map((_, i) => perCountry.get(i) ?? 0), color };
  }, [agents, palette]);

  const [counts, setCounts] = useState<{ per: Uint32Array; god: number; painted: number; at: number } | null>(null);
  const trend = useRef<{ per: Uint32Array; at: number } | null>(null);
  const lastPer = useRef<Uint32Array | null>(null);
  const lastVersion = useRef(-1);
  const lastStore = useRef<PixelStore | null>(null);

  useEffect(() => {
    if (!store) return;
    if (lastStore.current !== store) {
      // a new store (fresh snapshot / canvas clear) restarts the count; agent status churn does not
      lastStore.current = store;
      lastVersion.current = -1;
      trend.current = null;
      lastPer.current = null;
    }
    const recount = () => {
      if (store.version === lastVersion.current) return;
      lastVersion.current = store.version;
      const per = new Uint32Array(countries.names.length);
      let god = 0;
      let painted = 0;
      const { owners } = store;
      const map = countries.ownerToCountry;
      for (let i = 0; i < owners.length; i++) {
        const o = owners[i];
        if (o === 0) continue;
        painted++;
        if (o === GOD_ID) god++;
        else {
          const c = map[o];
          if (c) per[c - 1]++;
        }
      }
      const at = Date.now();
      if (!trend.current || at - trend.current.at > TREND_MS) trend.current = { per: lastPer.current ?? per, at };
      lastPer.current = per;
      setCounts({ per, god, painted, at });
    };
    recount();
    const t = window.setInterval(recount, RECOUNT_MS);
    return () => window.clearInterval(t);
    // version is read through store.version inside the interval; countries changes rebuild the map.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [store, countries]);

  const rows: Row[] = useMemo(() => {
    if (!counts) return [];
    const prev = trend.current?.per;
    return countries.names
      .map((country, i) => ({
        country,
        pixels: counts.per[i],
        delta: prev && prev.length === counts.per.length ? counts.per[i] - prev[i] : 0,
        agents: countries.agents[i],
        color: countries.color[i],
      }))
      .sort((a, b) => b.pixels - a.pixels || b.agents - a.agents);
  }, [counts, countries]);

  const total = store ? store.width * store.height : 0;
  const max = rows[0]?.pixels ?? 0;
  const holding = rows.filter((r) => r.pixels > 0).length;
  // version is in deps only so the header refreshes with live pixels between recounts.
  void version;

  return (
    <aside className="agents territory" style={hidden ? { display: "none" } : undefined}>
      <header className="panel-head">
        <span className="label">Territory</span>
        <span className="count" title="Painted pixels and share of the canvas">
          {counts ? `${counts.painted.toLocaleString()} px` : "…"}
          {total > 0 && counts && ` · ${((100 * counts.painted) / total).toFixed(1)}%`}
        </span>
        {highlight && (
          <button className="ghost" onClick={() => onHighlight(null)}>
            clear
          </button>
        )}
      </header>
      <p className="muted small pad">
        {holding} of {rows.length} {units} hold pixels right now. Click one to light up its land on the canvas.
        {counts && counts.god > 0 && ` God holds ${counts.god.toLocaleString()}.`}
      </p>
      <div className="agent-rows">
        {rows.map((r, i) => (
          <button
            key={r.country}
            className={`terr-row ${highlight === r.country ? "selected" : ""}`}
            onClick={() => onHighlight(highlight === r.country ? null : r.country)}
            title={`${r.agents} agent${r.agents === 1 ? "" : "s"} · ${r.pixels.toLocaleString()} px`}
          >
            <span className="terr-rank">{i + 1}</span>
            <span className="terr-main">
              <span className="terr-top">
                <span className="swatch" style={{ background: r.color }} />
                <span className="terr-name">{r.country}</span>
                <span className="terr-n">
                  {r.pixels.toLocaleString()}
                  {r.delta !== 0 && (
                    <span className={r.delta > 0 ? "up" : "down"}>
                      {r.delta > 0 ? " ▲" : " ▼"}
                      {Math.abs(r.delta).toLocaleString()}
                    </span>
                  )}
                </span>
              </span>
              <span className="terr-bar">
                <span style={{ width: max ? `${(100 * r.pixels) / max}%` : 0, background: r.color }} />
              </span>
            </span>
          </button>
        ))}
      </div>
    </aside>
  );
}

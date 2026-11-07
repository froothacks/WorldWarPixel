import { useMemo, useState } from "react";
import type { Agent } from "./api";

interface Props {
  agents: Agent[];
  palette: string[];
  selected: number | null;
  lastThought: Map<number, string>;
  onSelect: (id: number | null) => void;
}

/** Rows rendered at once; with thousands of agents the DOM gets heavy, so filter to reach the rest. */
const MAX_ROWS = 400;

export default function AgentList({ agents, palette, selected, lastThought, onSelect }: Props) {
  const [query, setQuery] = useState("");
  const matching = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q ? agents.filter((a) => a.name.toLowerCase().includes(q) || a.country.toLowerCase().includes(q)) : agents;
  }, [agents, query]);
  const shown = matching.length > MAX_ROWS ? matching.slice(0, MAX_ROWS) : matching;
  const active = agents.filter((a) => a.status !== "idle").length;

  return (
    <aside className="agents">
      <header className="panel-head">
        <span className="label">Agents</span>
        <span className="count">
          {agents.length}
          {active > 0 && ` · ${active} active`}
        </span>
      </header>
      <input className="search" placeholder="Filter by name or country" value={query} onChange={(e) => setQuery(e.target.value)} />
      <div className="agent-rows">
        {shown.map((a) => (
          <button
            key={a.id}
            className={`agent-row ${selected === a.id ? "selected" : ""}`}
            onClick={() => onSelect(selected === a.id ? null : a.id)}
          >
            <span className="swatch" style={{ background: palette[a.color] ?? "#888" }} />
            <span className="agent-main">
              <span className="agent-name">
                {a.name}
                <span className="agent-country">{a.country}</span>
              </span>
              <span className="agent-thought">{lastThought.get(a.id) ?? a.last_thought?.thought ?? "no turn yet"}</span>
            </span>
            <span className={`status status-${a.status}`}>{a.status}</span>
          </button>
        ))}
        {shown.length === 0 && <p className="muted small pad">No agents match.</p>}
        {matching.length > shown.length && (
          <p className="muted small pad">
            Showing {shown.length} of {matching.length.toLocaleString()}. Filter by name or country to find the rest.
          </p>
        )}
      </div>
    </aside>
  );
}

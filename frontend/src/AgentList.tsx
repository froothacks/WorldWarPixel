import { useMemo, useState } from "react";
import type { Agent } from "./api";

interface Props {
  agents: Agent[];
  palette: string[];
  selected: number | null;
  lastThought: Map<number, string>;
  onSelect: (id: number | null) => void;
}

export default function AgentList({ agents, palette, selected, lastThought, onSelect }: Props) {
  const [query, setQuery] = useState("");
  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q ? agents.filter((a) => a.name.toLowerCase().includes(q) || a.country.toLowerCase().includes(q)) : agents;
  }, [agents, query]);

  return (
    <aside className="agents">
      <input className="search" placeholder="Search agents or countries" value={query} onChange={(e) => setQuery(e.target.value)} />
      <div className="agent-rows">
        {shown.map((a) => (
          <button
            key={a.id}
            className={`agent-row ${selected === a.id ? "selected" : ""}`}
            onClick={() => onSelect(selected === a.id ? null : a.id)}
          >
            <span className="swatch" style={{ background: palette[a.color] ?? "#888" }} />
            <span className="agent-main">
              <span className="agent-name">{a.name}</span>
              <span className="muted small">{a.country} · {lastThought.get(a.id) ?? a.last_thought?.thought ?? "no turn yet"}</span>
            </span>
            <span className={`status status-${a.status}`}>{a.status}</span>
          </button>
        ))}
        {shown.length === 0 && <p className="muted small pad">No agents match.</p>}
      </div>
    </aside>
  );
}

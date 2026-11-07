import { useMemo, useState } from "react";
import type { Agent, Society } from "./api";

interface Props {
  agents: Agent[];
  societies?: Society[];
  palette: string[];
  selected: number | null;
  lastThought: Map<number, string>;
  onSelect: (id: number | null) => void;
}

export default function AgentList({ agents, societies = [], palette, selected, lastThought, onSelect }: Props) {
  const [query, setQuery] = useState("");
  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q ? agents.filter((a) => a.name.toLowerCase().includes(q) || a.country.toLowerCase().includes(q)) : agents;
  }, [agents, query]);
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
                {a.society != null && societies[a.society] && (
                  <span className="agent-society" style={{ color: societies[a.society].color }} title={`In ${societies[a.society].name}`}>
                    ● {societies[a.society].name}
                  </span>
                )}
              </span>
              <span className="agent-thought">{lastThought.get(a.id) ?? a.last_thought?.thought ?? "no turn yet"}</span>
            </span>
            <span className={`status status-${a.status}`}>{a.status}</span>
          </button>
        ))}
        {shown.length === 0 && <p className="muted small pad">No agents match.</p>}
      </div>
    </aside>
  );
}

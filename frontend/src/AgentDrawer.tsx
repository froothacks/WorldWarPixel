import { useEffect, useState } from "react";
import { api, type Agent, type AgentDetail, type Thought } from "./api";

interface Props {
  runId: string;
  agent: Agent;
  palette: string[];
  liveThoughts: Thought[];
  pixelsOwned: number;
  onClose: () => void;
}

export default function AgentDrawer({ runId, agent, palette, liveThoughts, pixelsOwned, onClose }: Props) {
  const [detail, setDetail] = useState<AgentDetail | null>(null);

  // Refetch whenever a new live thought arrives so position / history stay current.
  const liveCount = liveThoughts.length;
  const latestTs = liveThoughts[0]?.ts ?? 0;
  useEffect(() => {
    let cancelled = false;
    api.agent(runId, agent.id)
      .then((d) => !cancelled && setDetail(d))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [runId, agent.id, liveCount, latestTs]);

  const seen = new Set<number>();
  const fresh = detail?.id === agent.id ? detail : null;
  const history = fresh?.thoughts ?? [];
  // The snapshot's agent list is static; the detail fetch has where the agent is painting now.
  const pos = fresh ?? agent;
  const thoughts = [...liveThoughts, ...history]
    .filter((t) => (seen.has(t.ts) ? false : (seen.add(t.ts), true)))
    .sort((a, b) => b.ts - a.ts)
    .slice(0, 20);

  return (
    <aside className="drawer">
      <header className="drawer-header">
        <span className="swatch" style={{ background: palette[agent.color] ?? "#888" }} />
        <strong>{agent.name}</strong>
        <span className={`status status-${agent.status}`}>{agent.status}</span>
        <button className="ghost" onClick={onClose}>Close</button>
      </header>
      <dl className="facts">
        <dt>Country</dt><dd>{agent.country}</dd>
        <dt>Position</dt><dd>({pos.home_x}, {pos.home_y})</dd>
        <dt>Cooldown</dt><dd>{agent.cooldown}s</dd>
        <dt>Pixels held</dt><dd>{pixelsOwned.toLocaleString()}</dd>
      </dl>
      <p className="muted small">Its pixels are highlighted on the canvas. Persona and system prompt arrive with the LLM agents.</p>
      <h3>Thoughts</h3>
      <div className="thoughts">
        {thoughts.length === 0 && <p className="muted small">No turns yet.</p>}
        {thoughts.map((t) => (
          <div key={t.ts} className="thought">
            <span className="muted small">{new Date(t.ts * 1000).toLocaleTimeString()}</span>
            <p>{t.thought}</p>
          </div>
        ))}
      </div>
    </aside>
  );
}

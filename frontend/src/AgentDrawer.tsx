import { Fragment, useEffect, useState } from "react";
import { api, type Agent, type AgentDetail, type Thought } from "./api";

interface Props {
  runId: string;
  agent: Agent;
  palette: string[];
  liveThoughts: Thought[];
  pixelsOwned: number;
  onClose: () => void;
}

const CARD_FIELDS: [string, string][] = [
  ["age", "Age"],
  ["sex", "Sex"],
  ["ethnic_group", "Ethnic group"],
  ["settlement", "Lives in"],
  ["marital_status", "Marital status"],
  ["children", "Children"],
  ["education", "Education"],
  ["employment", "Employment"],
  ["occupation", "Occupation"],
  ["hobbies", "Hobbies"],
  ["income_decile", "Income decile (1–10)"],
  ["social_class", "Social class"],
  ["religion", "Religion"],
  ["religiosity", "Religiosity"],
  ["attendance", "Attends services"],
  ["god_importance", "Importance of God (1–10)"],
  ["left_right", "Left–right (1–10)"],
  ["political_interest", "Politics"],
  ["trust", "Trust"],
  ["democracy_importance", "Democracy matters (1–10)"],
  ["democratic_system", "Democratic system"],
  ["strong_leader", "Strong leader"],
  ["income_equality", "Equality vs incentives (1–10)"],
  ["env_vs_growth", "Environment vs growth"],
  ["happiness", "Happiness"],
  ["life_satisfaction", "Life satisfaction (1–10)"],
  ["national_pride", "National pride"],
];

const CONFIDENCE: Record<string, string> = {
  high: "Profile match: confident",
  medium: "Profile match: likely",
  low: "Profile match: uncertain (little public info)",
};

function asList(v: unknown): string[] {
  return Array.isArray(v) ? v.map(String) : [];
}

function linkLabel(url: string): string {
  return url.replace(/^https?:\/\/(www\.)?/, "").replace(/\/$/, "");
}

function formatValue(v: string | number | string[] | null): string {
  return Array.isArray(v) ? v.join(", ") : String(v);
}

export default function AgentDrawer({ runId, agent, palette, liveThoughts, pixelsOwned, onClose }: Props) {
  const [detail, setDetail] = useState<AgentDetail | null>(null);
  const [failedId, setFailedId] = useState<number | null>(null);

  // Refetch whenever a new live thought arrives so position / history stay current.
  const liveCount = liveThoughts.length;
  const latestTs = liveThoughts[0]?.ts ?? 0;
  useEffect(() => {
    let cancelled = false;
    api.agent(runId, agent.id)
      .then((d) => !cancelled && setDetail(d))
      .catch(() => !cancelled && setFailedId(agent.id));
    return () => {
      cancelled = true;
    };
  }, [runId, agent.id, liveCount, latestTs]);

  const seen = new Set<number>();
  const fresh = detail?.id === agent.id ? detail : null;
  const failed = !fresh && failedId === agent.id;
  const history = fresh?.thoughts ?? [];
  // The snapshot's agent list is static; the detail fetch has where the agent is painting now.
  const pos = fresh ?? agent;
  const card = fresh?.persona_card ?? {};
  const person = card.source === "roster";
  const links = [...(typeof card.linkedin === "string" ? [card.linkedin] : []), ...asList(card.links)];
  const thoughts = [...liveThoughts, ...history]
    .filter((t) => (seen.has(t.ts) ? false : (seen.add(t.ts), true)))
    .sort((a, b) => b.ts - a.ts)
    .slice(0, 20);

  return (
    <aside className="drawer">
      <header className="panel-head">
        <span className="swatch" style={{ background: palette[agent.color] ?? "#888" }} />
        <strong>{agent.name}</strong>
        <span className={`status status-${agent.status}`}>{agent.status}</span>
        <button className="ghost" onClick={onClose}>
          ← all agents
        </button>
      </header>
      <div className="drawer-body">
        <dl className="facts">
          {person ? (
            <>
              <dt>Affiliation</dt><dd>{String(card.affiliation ?? "unknown")}</dd>
            </>
          ) : (
            <>
              <dt>Country</dt><dd>{agent.country}</dd>
            </>
          )}
          <dt>Position</dt><dd>({pos.home_x}, {pos.home_y})</dd>
          <dt>Cooldown</dt><dd>{agent.cooldown}s</dd>
          <dt>Pixels held</dt><dd>{pixelsOwned.toLocaleString()}</dd>
        </dl>
        {person && (
          <section className="profile">
            <p className="profile-headline">{String(card.headline ?? "")}</p>
            {asList(card.highlights).length > 0 && (
              <ul className="profile-highlights">
                {asList(card.highlights).map((h) => (
                  <li key={h}>{h}</li>
                ))}
              </ul>
            )}
            {links.length > 0 && (
              <div className="profile-links">
                {links.map((url) => (
                  <a key={url} href={url} target="_blank" rel="noreferrer">
                    {url.includes("linkedin.com/in/") ? "LinkedIn" : linkLabel(url)}
                  </a>
                ))}
              </div>
            )}
            <p className="faint small">{CONFIDENCE[String(card.confidence)] ?? ""}</p>
          </section>
        )}
        <p className="small persona">{fresh?.persona ?? (failed ? "Couldn't load this agent's persona." : "Loading persona…")}</p>
        {!person && Object.keys(card).length > 0 && (
          <details className="prompt">
            <summary className="muted small">
              Persona card · {card.source === "wvs7" ? "WVS Wave 7 respondent" : "synthetic respondent"}
              {card.survey_country && card.survey_country !== agent.country
                ? ` (proxy: ${card.survey_country})`
                : ""}
            </summary>
            <dl className="facts small">
              {CARD_FIELDS.filter(([key]) => card[key] != null && card[key] !== "").map(
                ([key, label]) => (
                  <Fragment key={key}>
                    <dt>{label}</dt>
                    <dd>{formatValue(card[key])}</dd>
                  </Fragment>
                ),
              )}
            </dl>
          </details>
        )}
        <p className="faint small">Pixels highlighted on the canvas.</p>
        {fresh?.system_prompt && (
          <details className="prompt">
            <summary className="muted small">System prompt</summary>
            <pre>{fresh.system_prompt}</pre>
          </details>
        )}
        <h3>Thoughts</h3>
        <div className="thoughts">
          {thoughts.length === 0 && <p className="muted small">No turns yet.</p>}
          {thoughts.map((t) => (
            <div key={t.ts} className="thought">
              <span className="faint small mono">{new Date(t.ts * 1000).toLocaleTimeString([], { hour12: false })}</span>
              <p>{t.thought}</p>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}

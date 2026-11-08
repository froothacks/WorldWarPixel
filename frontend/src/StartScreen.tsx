import { useEffect, useState } from "react";
import { api, DEFAULT_CONFIG, type ModelInfo, type RosterSummary, type RunConfig, type RunSummary } from "./api";
import PixelField from "./PixelField";
import { InlineError } from "./Errors";

interface Props {
  onStarted: (run: RunSummary) => void;
}

const SIZES: [number, number][] = [
  [256, 160],
  [512, 320],
  [768, 480],
  [1024, 640],
  [512, 512],
];

const SLIDER_MAX = 2000;
const MAX_AGENTS = 10000;
/** Random turns are instant, so past this the single-core event loop saturates. */
const RANDOM_TURNS_PER_SEC = 100;
const PRESETS = [50, 100, 250, 500, 1000, 2000, 5000];

const BRAINS: { value: RunConfig["policy"]; title: string; hint: string }[] = [
  { value: "llm", title: "Free-for-all", hint: "Language model, no steering. Each turn is one call to the configured model." },
  {
    value: "llm-guided",
    title: "Guided",
    hint: "Language model with event framing, examples and free-spot hints. One call per turn.",
  },
  { value: "random", title: "Random", hint: "No model calls. Agents paint random blobs. Good for testing." },
];

function ago(ts: number): string {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export default function StartScreen({ onStarted }: Props) {
  const [cfg, setCfg] = useState<RunConfig>(DEFAULT_CONFIG);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [existing, setExisting] = useState<RunSummary[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [rosters, setRosters] = useState<RosterSummary[]>([]);

  useEffect(() => {
    api.listRuns().then(setExisting).catch(() => setExisting([]));
    api
      .listModels()
      .then((ms) => {
        setModels(ms);
        // Fall back to the first available preset when the current choice is unusable
        // (e.g. a Modal-only server where the gpt-6-luna default has no OpenAI key).
        setCfg((c) => {
          if (c.policy === "random" || ms.length === 0) return c;
          if (ms.some((m) => m.id === c.model && m.available)) return c;
          const first = ms.find((m) => m.available);
          return first ? { ...c, model: first.id } : c;
        });
      })
      .catch(() => setModels([]));
    api.rosters().then(setRosters).catch(() => setRosters([]));
  }, []);

  const set = <K extends keyof RunConfig>(key: K, value: RunConfig[K]) =>
    setCfg((c) => ({ ...c, [key]: value }));

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      onStarted(await api.createRun(cfg));
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  };

  // With premoving an agent's next turn starts when ~1 s of its queued pixels remain, so the
  // real period is the longer of the cooldown and the time its batch takes to drain.
  const roster = rosters.find((r) => r.id === cfg.roster);
  const agents = roster ? roster.size : cfg.agents;
  const period = Math.max(cfg.cooldown, cfg.pixels_per_turn * cfg.pixel_interval - 1);
  const turnsPerMin = Math.round((agents * 60) / period);
  const pxPerMin = turnsPerMin * cfg.pixels_per_turn;
  const brain = BRAINS.find((b) => b.value === cfg.policy) ?? BRAINS[0];
  const turnsPerSec = agents / period;
  const warning =
    cfg.policy === "random" && turnsPerSec > RANDOM_TURNS_PER_SEC
      ? `${Math.round(turnsPerSec)} random turns a second will saturate one CPU core; raise the cooldown or use fewer agents.`
      : cfg.policy !== "random" && turnsPerSec > cfg.max_concurrency
        ? `Turns are capped by "Turns in flight" (${cfg.max_concurrency}); with ${agents.toLocaleString()} agents each one will wait longer than its ${cfg.cooldown}s cooldown between model calls.`
        : null;
  const runs = [...existing].sort((a, b) => b.created_at - a.created_at);

  return (
    <div className="start">
      <aside className="start-hero">
        <PixelField />
        <div className="hero-copy">
          <h1>World War Pixel</h1>
          <p>
            Give a crowd of agents one shared canvas and one message board, then see what they draw, who they team up
            with and what they fight over.
          </p>
          <dl className="hero-stats">
            <div>
              <dt>Agents</dt>
              <dd>{agents.toLocaleString()}</dd>
            </div>
            <div>
              <dt>Canvas</dt>
              <dd>
                {cfg.width}×{cfg.height}
              </dd>
            </div>
            <div>
              <dt>Turns / min</dt>
              <dd>{turnsPerMin.toLocaleString()}</dd>
            </div>
          </dl>
        </div>
      </aside>

      <main className="start-main">
        <section className="setup">
          <h2>New simulation</h2>

          <div className="field">
            <span className="field-label">Brain</span>
            <div className="segmented">
              {BRAINS.map((b) => (
                <button key={b.value} type="button" className={cfg.policy === b.value ? "on" : ""} onClick={() => set("policy", b.value)}>
                  {b.title}
                </button>
              ))}
            </div>
            <span className="hint">{brain.hint}</span>
          </div>

          {cfg.policy !== "random" && models.length > 0 && (
            <div className="field">
              <span className="field-label">Model</span>
              <select value={cfg.model} onChange={(e) => set("model", e.target.value)}>
                {cfg.model && !models.some((m) => m.id === cfg.model) && (
                  <option value={cfg.model}>{cfg.model}</option>
                )}
                {models.map((m) => (
                  <option key={m.id} value={m.id} disabled={!m.available}>
                    {m.label}
                    {m.available ? "" : " — unavailable"}
                  </option>
                ))}
              </select>
              <span className="hint">Every agent and the narrator use this model for the whole run.</span>
            </div>
          )}

          {rosters.length > 0 && (
            <div className="field">
              <span className="field-label">Players</span>
              <div className="segmented">
                <button type="button" className={cfg.roster === "world" ? "on" : ""} onClick={() => set("roster", "world")}>
                  The world
                </button>
                {rosters.map((r) => (
                  <button key={r.id} type="button" className={cfg.roster === r.id ? "on" : ""} onClick={() => set("roster", r.id)}>
                    {r.title} · {r.size}
                  </button>
                ))}
              </div>
              <span className="hint">
                {roster
                  ? `One agent per attendee of ${roster.event}, each playing as themselves with a persona from their public profile. No teams: any alliances form on the board.`
                  : "Agents are seated per country by world population."}
              </span>
            </div>
          )}

          {!roster && (
          <label className="field">
            <span className="field-label">Agents</span>
            <div className="agent-count">
              <input
                type="range"
                min={1}
                max={SLIDER_MAX}
                value={Math.min(cfg.agents, SLIDER_MAX)}
                onChange={(e) => set("agents", +e.target.value)}
              />
              <input
                type="number"
                min={1}
                max={MAX_AGENTS}
                value={cfg.agents}
                onChange={(e) => set("agents", Math.max(1, Math.min(MAX_AGENTS, Math.round(+e.target.value) || 1)))}
              />
            </div>
            <div className="presets">
              {PRESETS.map((n) => (
                <button key={n} type="button" className={cfg.agents === n ? "on" : ""} onClick={() => set("agents", n)}>
                  {n.toLocaleString()}
                </button>
              ))}
            </div>
            <span className="hint">Up to {MAX_AGENTS.toLocaleString()}. Agents are seated per country by population, so more agents means more countries show up.</span>
          </label>
          )}

          <div className="field">
            <span className="field-label">Canvas</span>
            <div className="sizes">
              {SIZES.map(([w, h]) => {
                const on = cfg.width === w && cfg.height === h;
                return (
                  <button
                    key={`${w}x${h}`}
                    type="button"
                    className={`size ${on ? "on" : ""}`}
                    onClick={() => {
                      set("width", w);
                      set("height", h);
                    }}
                  >
                    <span className="size-box" style={{ width: (w / 1024) * 36, height: (h / 1024) * 36 }} />
                    <span>
                      {w}×{h}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="field">
            <span className="field-label">Societies</span>
            <div className="segmented">
              {[0, 2, 3, 4].map((n) => (
                <button key={n} type="button" className={cfg.societies === n ? "on" : ""} onClick={() => set("societies", n)}>
                  {n === 0 ? "Off" : n}
                </button>
              ))}
            </div>
            <span className="hint">
              {cfg.societies
                ? "Agents are split into groups, each with its own board. They can peek at other groups and move. Each group has a screen that listens to people in the room."
                : "One shared board for everyone."}
            </span>
            {cfg.societies > 0 && (
              <input
                type="text"
                className="society-theme"
                maxLength={300}
                value={cfg.society_theme ?? ""}
                placeholder="Optional: what the group names should be about (e.g. deep-sea creatures)"
                onChange={(e) => set("society_theme", e.target.value || null)}
              />
            )}
          </div>

          <details className="advanced">
            <summary>Advanced</summary>
            <div className="advanced-grid">
              <label>
                Cooldown (s)
                <input type="number" min={0.1} step={0.5} value={cfg.cooldown} onChange={(e) => set("cooldown", +e.target.value)} />
              </label>
              <label>
                Pixels per turn
                <input type="number" min={1} max={1024} value={cfg.pixels_per_turn} onChange={(e) => set("pixels_per_turn", +e.target.value)} />
              </label>
              <label title="Milliseconds between an agent's successive pixels; minimum 10 ms">
                Pixel gap (ms)
                <input type="number" min={10} max={5000} step={5} value={Math.round(cfg.pixel_interval * 1000)} onChange={(e) => set("pixel_interval", +e.target.value / 1000)} />
              </label>
              <label>
                Posts per turn
                <input type="number" min={0} max={10} value={cfg.posts_per_turn} onChange={(e) => set("posts_per_turn", +e.target.value)} />
              </label>
              <label>
                Turns in flight
                <input type="number" min={1} max={1000} value={cfg.max_concurrency} onChange={(e) => set("max_concurrency", +e.target.value)} />
              </label>
              <label>
                Narrator every (s)
                <input type="number" min={5} max={3600} step={5} value={cfg.narrator_interval} onChange={(e) => set("narrator_interval", +e.target.value)} />
              </label>
              <label>
                Seed
                <input type="number" value={cfg.seed ?? ""} placeholder="random" onChange={(e) => set("seed", e.target.value === "" ? null : +e.target.value)} />
              </label>
            </div>
          </details>

          <div className="launch">
            <span className="estimate">
              About {turnsPerMin.toLocaleString()} turns and {pxPerMin.toLocaleString()} pixels a minute
            </span>
            <button className="primary" onClick={start} disabled={busy}>
              {busy ? "Starting…" : "Start simulation"}
            </button>
          </div>
          {warning && <p className="warn">{warning}</p>}
          {error && <InlineError message={error} block />}
        </section>

        <section className="recent">
          <h2>
            Recent runs <span className="count">{runs.length || ""}</span>
          </h2>
          {runs.length === 0 ? (
            <p className="faint">Runs you start will show up here.</p>
          ) : (
            <ul className="run-list">
              {runs.map((r) => (
                <li key={r.id}>
                  <button onClick={() => onStarted(r)}>
                    <span className={`state state-${r.state}`}>{r.state}</span>
                    <span className="mono run-id">{r.id}</span>
                    <span className="run-meta">
                      {r.agents} {r.config.policy === "random" ? "random" : (r.config.model ?? "LLM")} agents · {r.config.width}×{r.config.height} ·{" "}
                      {r.stats.pixels.toLocaleString()} px
                    </span>
                    <span className="run-when">{ago(r.created_at)}</span>
                    <span className="run-open">Open →</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>
      </main>
    </div>
  );
}

import { useEffect, useState } from "react";
import { api, DEFAULT_CONFIG, type RunConfig, type RunSummary } from "./api";

interface Props {
  onStarted: (run: RunSummary) => void;
}

const SIZES = [128, 256, 512, 1024];

export default function StartScreen({ onStarted }: Props) {
  const [cfg, setCfg] = useState<RunConfig>(DEFAULT_CONFIG);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [existing, setExisting] = useState<RunSummary[]>([]);

  useEffect(() => {
    api.listRuns().then(setExisting).catch(() => setExisting([]));
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

  const callsPerMin = Math.round((cfg.agents * 60) / cfg.cooldown);
  const pxPerMin = callsPerMin * cfg.pixels_per_turn;

  return (
    <div className="start">
      <header className="start-header">
        <h1>WorldWarPixel</h1>
        <p>A swarm of agents sharing one canvas and one message board.</p>
      </header>
      <div className="start-grid">
        <section className="panel">
          <h2>New run</h2>
          <label>
            Agents brain
            <select value={cfg.policy} onChange={(e) => set("policy", e.target.value as RunConfig["policy"])}>
              <option value="llm">LLM (configured endpoint)</option>
              <option value="random">Random walk (no LLM, free)</option>
            </select>
          </label>
          <label>
            Agents
            <input type="number" min={1} max={5000} value={cfg.agents} onChange={(e) => set("agents", +e.target.value)} />
          </label>
          <label>
            Canvas
            <select value={cfg.width} onChange={(e) => { set("width", +e.target.value); set("height", +e.target.value); }}>
              {SIZES.map((s) => (
                <option key={s} value={s}>{s} × {s}</option>
              ))}
            </select>
          </label>
          <label>
            Cooldown (s)
            <input type="number" min={0.1} step={0.5} value={cfg.cooldown} onChange={(e) => set("cooldown", +e.target.value)} />
          </label>
          <label>
            Pixels per turn
            <input type="number" min={1} max={256} value={cfg.pixels_per_turn} onChange={(e) => set("pixels_per_turn", +e.target.value)} />
          </label>
          <label>
            Posts per turn
            <input type="number" min={0} max={10} value={cfg.posts_per_turn} onChange={(e) => set("posts_per_turn", +e.target.value)} />
          </label>
          <label>
            Max turns in flight
            <input type="number" min={1} max={500} value={cfg.max_concurrency} onChange={(e) => set("max_concurrency", +e.target.value)} />
          </label>
          <label>
            Narrator update (s)
            <input type="number" min={5} max={3600} step={5} value={cfg.narrator_interval} onChange={(e) => set("narrator_interval", +e.target.value)} />
          </label>
          <label>
            Seed (optional)
            <input type="number" value={cfg.seed ?? ""} placeholder="random" onChange={(e) => set("seed", e.target.value === "" ? null : +e.target.value)} />
          </label>
          <p className="muted">
            About {callsPerMin.toLocaleString()} turns/min and up to {pxPerMin.toLocaleString()} px/min.
            {cfg.policy === "llm" ? " Each turn is one model call." : " No model calls; agents paint random blobs."}
          </p>
          {error && <p className="error">{error}</p>}
          <button className="primary" onClick={start} disabled={busy}>
            {busy ? "Starting…" : "Start simulation"}
          </button>
        </section>
        <section className="panel">
          <h2>Runs on this server</h2>
          {existing.length === 0 && <p className="muted">None yet.</p>}
          <ul className="run-list">
            {existing.map((r) => (
              <li key={r.id}>
                <button onClick={() => onStarted(r)}>
                  <span className={`state state-${r.state}`}>{r.state}</span>
                  <span className="mono">{r.id}</span>
                  <span className="muted">
                    {r.config.width}×{r.config.height} · {r.agents} agents · {r.stats.pixels.toLocaleString()} px
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}

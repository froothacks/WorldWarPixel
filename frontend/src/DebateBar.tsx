import { useEffect, useState } from "react";
import { api, type Agent, type StageState } from "./api";

interface Props {
  runId: string;
  state: StageState | null;
  agents: Agent[];
  palette: string[];
  sound: boolean;
  onSound: (enabled: boolean) => void;
  onSelectAgent: (id: number) => void;
  onState: (state: StageState) => void;
}

export default function DebateBar({
  runId,
  state,
  agents,
  palette,
  sound,
  onSound,
  onSelectAgent,
  onState,
}: Props) {
  const [now, setNow] = useState(() => Date.now() / 1000);
  const [error, setError] = useState<string | null>(null);
  const phase = state?.phase ?? "idle";
  const debaters = state?.debaters ?? [];
  const audienceUrl = `${location.origin}${location.pathname}?audience=${runId}`;

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const act = async (request: () => Promise<StageState>) => {
    setError(null);
    try {
      onState(await request());
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const copyAudience = async () => {
    try {
      await navigator.clipboard.writeText(audienceUrl);
    } catch {
      window.prompt("Copy the audience link", audienceUrl);
    }
  };
  const votes = (id: number) => state?.votes[String(id)] ?? 0;
  const voteLeft =
    state?.vote_ends_at === null || state?.vote_ends_at === undefined
      ? null
      : Math.max(0, Math.ceil(state.vote_ends_at - now));
  const holdAgent = state?.hold
    ? agents.find((agent) => agent.id === state.hold?.agent_id)
    : undefined;
  const holdSecondsLeft =
    state?.hold && state.hold.until > now
      ? Math.max(0, Math.ceil(state.hold.until - now))
      : null;

  return (
    <div className="debate-bar">
      <div className="debate-controls">
        {phase === "idle" ? (
          <>
            <button className="primary" onClick={() => void act(() => api.stageStart(runId))}>
              Start debate
            </button>
            <button
              className={state?.auto ? "on" : ""}
              onClick={() => void act(() => api.stageAuto(runId, !state?.auto))}
            >
              Auto {state?.auto ? "on" : "off"}
            </button>
          </>
        ) : (
          <>
            <span className={`debate-phase stage-phase-${phase}`}>{phase}</span>
            <span className="debate-topic" title={state?.topic}>{state?.topic}</span>
            {debaters.map((debater) => (
              <button
                key={debater.id}
                className="debate-chip"
                onClick={() => onSelectAgent(debater.id)}
                title={`Select ${debater.name}`}
              >
                <span className="debate-swatch" style={{ background: palette[debater.color] ?? debater.color_hex }} />
                {debater.name}
              </button>
            ))}
            {debaters.length === 2 && (
              <span className="debate-votes">
                {votes(debaters[0].id)} – {votes(debaters[1].id)}
              </span>
            )}
            {phase === "voting" && voteLeft !== null && (
              <span className="debate-countdown">{voteLeft}s</span>
            )}
            <button className="danger" onClick={() => void act(() => api.stageStop(runId))}>
              End
            </button>
          </>
        )}
      </div>
      <div className="debate-links">
        <button className={sound ? "on" : ""} onClick={() => onSound(!sound)}>
          Sound {sound ? "on" : "off"}
        </button>
        <a className="button-link" href={`?stage=${runId}`} target="_blank" rel="noreferrer">
          Presenter
        </a>
        <a className="button-link" href={`?audience=${runId}`} target="_blank" rel="noreferrer">
          Audience
        </a>
        <button className="ghost" onClick={() => void copyAudience()} title="Copy audience link">
          Copy
        </button>
        {[0, 1].map((podium) => {
          const count = state?.devices[String(podium)] ?? 0;
          return (
            <span className="debate-device" key={podium} title={`${count} connected`}>
              Podium {podium + 1} <strong className={count ? "connected" : ""}>{count ? "●" : "○"}</strong>
              {count > 1 && ` ${count}`}
            </span>
          );
        })}
        {state?.hold && holdSecondsLeft !== null && (
          <span className="debate-hold">
            {holdAgent?.name ?? `Agent ${state.hold.agent_id}`} holds region · {holdSecondsLeft}s
          </span>
        )}
      </div>
      {error && <span className="debate-error" title={error}>{error}</span>}
    </div>
  );
}

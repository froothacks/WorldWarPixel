import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import { api, stageAudioUrl, type Debater, type StageLine, type StageReaction, type StageState } from "./api";
import { openStageStream, type StageEvent } from "./ws";

interface Props {
  runId: string;
  mode: "stage" | "audience";
  /** Show and play a single podium only: a browser stand-in for one ESP32 device. */
  podium: number | null;
}

const REACTIONS: { key: StageReaction; label: string }[] = [
  { key: "cheer", label: "Cheer" },
  { key: "boo", label: "Boo" },
  { key: "laugh", label: "Laugh" },
  { key: "gasp", label: "Gasp" },
];
const REACTION_MARK: Record<StageReaction, string> = { cheer: "!!", boo: "boo", laugh: "ha", gasp: "oh" };
const TRANSCRIPT_MAX = 200;

function voterId(): string {
  let id = localStorage.getItem("wwp-voter");
  if (!id) {
    id = Math.random().toString(36).slice(2, 12);
    localStorage.setItem("wwp-voter", id);
  }
  return id;
}

function initials(name: string): string {
  return name
    .split(/\s+/)
    .map((w) => w[0] ?? "")
    .join("")
    .slice(0, 2)
    .toUpperCase();
}

function browserVoice(line: StageLine): SpeechSynthesisUtterance {
  const u = new SpeechSynthesisUtterance(line.text);
  const voices = speechSynthesis.getVoices().filter((v) => v.lang.startsWith("en"));
  if (voices.length) u.voice = voices[Math.abs(line.speaker_id) % voices.length];
  u.pitch = line.kind === "narrator" ? 1 : 0.8 + (Math.abs(line.speaker_id) % 5) * 0.1;
  u.rate = 1.05;
  return u;
}

/** Plays lines one after another; the server already paces them, this just avoids overlap. */
function usePlayer(runId: string, enabled: boolean) {
  const queue = useRef<StageLine[]>([]);
  const busy = useRef(false);
  const audio = useRef<HTMLAudioElement | null>(null);
  const [speaking, setSpeaking] = useState<StageLine | null>(null);

  const play = useCallback(
    (line: StageLine, done: () => void) => {
      const hasSpeech = typeof speechSynthesis !== "undefined";
      if (!enabled) {
        window.setTimeout(done, line.duration * 1000);
      } else if (line.audio) {
        const el = new Audio(stageAudioUrl(runId, line.id));
        audio.current = el;
        el.onended = done;
        el.onerror = done;
        el.play().catch(done);
      } else if (hasSpeech) {
        const u = browserVoice(line);
        u.onend = done;
        u.onerror = done;
        speechSynthesis.speak(u);
      } else {
        window.setTimeout(done, line.duration * 1000);
      }
    },
    [enabled, runId],
  );

  const next = useCallback(
    function advance() {
      const line = queue.current.shift();
      if (!line) {
        busy.current = false;
        setSpeaking(null);
        return;
      }
      busy.current = true;
      setSpeaking(line);
      play(line, advance);
    },
    [play],
  );

  const push = useCallback(
    (line: StageLine) => {
      // Never fall more than one line behind the live debate.
      queue.current = [...queue.current.slice(-1), line];
      if (!busy.current) next();
    },
    [next],
  );

  useEffect(
    () => () => {
      audio.current?.pause();
      if (typeof speechSynthesis !== "undefined") speechSynthesis.cancel();
    },
    [],
  );
  return { push, speaking };
}

export default function StageView({ runId, mode, podium }: Props) {
  const [state, setState] = useState<StageState | null>(null);
  const [transcript, setTranscript] = useState<StageLine[]>([]);
  const [live, setLive] = useState(false);
  const [sound, setSound] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [myVote, setMyVote] = useState<number | null>(null);
  const [verdict, setVerdict] = useState<{ winner: number | null; debate: number } | null>(null);
  const [bursts, setBursts] = useState<{ id: number; reaction: StageReaction; agent: number | null }[]>([]);
  const [now, setNow] = useState(() => Date.now() / 1000);
  const { push, speaking } = usePlayer(runId, sound && mode === "stage");
  const burstId = useRef(0);
  const logRef = useRef<HTMLDivElement>(null);

  const handle = useCallback(
    (ev: StageEvent) => {
      switch (ev.type) {
        case "stage.snapshot": {
          const { type: _t, transcript: lines, ...rest } = ev;
          setState(rest);
          setTranscript((lines ?? []).slice(-TRANSCRIPT_MAX));
          break;
        }
        case "stage.state": {
          const { type: _t, ...rest } = ev;
          setState((prev) => {
            if (prev && prev.debate_id !== rest.debate_id) {
              setMyVote(null);
              setVerdict(null);
            }
            return rest;
          });
          break;
        }
        case "stage.line": {
          const { type: _t, ...line } = ev;
          setTranscript((t) => [...t, line].slice(-TRANSCRIPT_MAX));
          if (podium === null || line.podium === podium || line.podium === null) push(line);
          break;
        }
        case "stage.votes":
          setState((prev) => (prev ? { ...prev, votes: ev.votes } : prev));
          break;
        case "stage.verdict":
          setVerdict({ winner: ev.winner_id, debate: ev.debate_id });
          setState((prev) => (prev ? { ...prev, votes: ev.votes } : prev));
          break;
        case "stage.reaction": {
          const id = ++burstId.current;
          setBursts((b) => [...b.slice(-12), { id, reaction: ev.reaction, agent: ev.agent_id }]);
          window.setTimeout(() => setBursts((b) => b.filter((x) => x.id !== id)), 1800);
          break;
        }
      }
    },
    [podium, push],
  );

  useEffect(() => openStageStream(runId, handle, setLive), [runId, handle]);

  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now() / 1000), 500);
    return () => window.clearInterval(t);
  }, []);

  useEffect(() => {
    document.documentElement.classList.add("stage-page");
    return () => document.documentElement.classList.remove("stage-page");
  }, []);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [transcript.length]);

  const act = async (fn: () => Promise<unknown>) => {
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const vote = (agentId: number) =>
    act(async () => {
      await api.stageVote(runId, voterId(), agentId);
      setMyVote(agentId);
    });

  const phase = state?.phase ?? "idle";
  const debaters = state?.debaters ?? [];
  const shown = podium === null ? debaters : debaters.filter((d) => d.podium === podium);
  const narratorLine = speaking && speaking.kind !== "debate" ? speaking : null;
  const voteLeft = state?.vote_ends_at ? Math.max(0, Math.ceil(state.vote_ends_at - now)) : null;
  const audienceUrl = `${location.origin}${location.pathname}?audience=${runId}`;

  return (
    <div className={`stage stage-${mode}${podium !== null ? " stage-single" : ""}`}>
      <header className="stage-top">
        <a className="brand" href={`?run=${runId}`}>
          WorldWarPixel
        </a>
        <span className="crumb">
          <span className="slash">/</span>
          <span>{mode === "audience" ? "audience" : podium !== null ? `podium ${podium + 1}` : "live debate"}</span>
          <span className={`state stage-phase-${phase}`}>{phase}</span>
        </span>
        <span className="spacer" />
        <span className={`conn ${live ? "conn-on" : ""}`}>{live ? "● live" : "○ connecting"}</span>
        {mode === "stage" && (
          <div className="actions">
            <button
              className={sound ? "on" : ""}
              onClick={() => setSound(!sound)}
              title={state?.speech ? "Agent voices (server TTS)" : "Agent voices (browser speech; no TTS key on the server)"}
            >
              {sound ? "Sound on" : "Enable sound"}
            </button>
            {podium === null && (
              <>
                <button
                  className={state?.auto ? "on" : ""}
                  onClick={() => act(() => api.stageAuto(runId, !state?.auto))}
                  title="Start a new debate between the fiercest rivals whenever the stage is idle"
                >
                  Auto
                </button>
                {phase === "idle" ? (
                  <button className="primary" onClick={() => act(() => api.stageStart(runId))}>
                    Start debate
                  </button>
                ) : (
                  <button className="danger" onClick={() => act(() => api.stageStop(runId))}>
                    End debate
                  </button>
                )}
              </>
            )}
          </div>
        )}
      </header>

      {error && <div className="stage-error">{error}</div>}

      <section className="stage-floor">
        <div className="stage-topic">
          {phase === "idle" ? (
            <span className="muted">
              {verdict ? "Debate over." : "The stage is empty."}{" "}
              {mode === "stage" ? "Start a debate to bring the two fiercest rivals to the podiums." : "Waiting for the next debate…"}
            </span>
          ) : (
            <>
              <span className="stage-topic-label">Debate #{state?.debate_id}</span>
              <span className="stage-topic-text">{state?.topic}</span>
              {state?.region && (
                <span className="mono faint">
                  region {state.region.x},{state.region.y} {state.region.w}×{state.region.h}
                </span>
              )}
            </>
          )}
        </div>

        <div className="podiums">
          {shown.map((d, i) => (
            <Podium
              key={d.id}
              debater={d}
              line={speaking?.speaker_id === d.id && speaking.kind === "debate" ? speaking : null}
              last={[...transcript].reverse().find((l) => l.speaker_id === d.id && l.debate_id === state?.debate_id)}
              votes={state?.votes[String(d.id)] ?? 0}
              total={Object.values(state?.votes ?? {}).reduce((a, b) => a + b, 0)}
              winner={verdict?.debate === state?.debate_id && verdict?.winner === d.id}
              bursts={bursts.filter((b) => b.agent === d.id)}
              side={i === 0 ? "left" : "right"}
            />
          ))}
          {podium === null && debaters.length === 2 && <div className="vs">VS</div>}
        </div>

        <div className={`narrator-line ${narratorLine ? "on" : ""}`}>
          {narratorLine && (
            <>
              <span className="narrator-name">{narratorLine.kind === "verdict" ? "Verdict" : narratorLine.name}</span>
              <span>{narratorLine.text}</span>
            </>
          )}
        </div>

        <div className="crowd">
          {bursts
            .filter((b) => b.agent === null)
            .map((b) => (
              <span key={b.id} className={`burst burst-${b.reaction}`}>
                {REACTION_MARK[b.reaction]}
              </span>
            ))}
        </div>
      </section>

      {podium === null && (
        <section className="stage-bottom">
          <div className="stage-panel transcript" ref={logRef}>
            <div className="panel-head">
              <span>Transcript</span>
              <span className="faint mono">{transcript.length}</span>
            </div>
            {transcript.map((l) => (
              <div key={l.id} className={`tline tline-${l.kind}`}>
                <span className="dot" style={{ background: l.color_hex ?? "var(--accent)" }} />
                <span className="tline-name">{l.kind === "debate" ? l.name : l.kind === "verdict" ? "Verdict" : l.name}</span>
                {l.emotion && l.kind === "debate" && <span className="faint">({l.emotion})</span>}
                <span className="tline-text">{l.text}</span>
              </div>
            ))}
            {transcript.length === 0 && <p className="muted pad">Nothing said yet.</p>}
          </div>

          <div className="stage-panel audience-panel">
            <div className="panel-head">
              <span>{phase === "voting" ? `Vote now${voteLeft !== null ? ` · ${voteLeft}s` : ""}` : "Audience"}</span>
            </div>
            <div className="vote-row">
              {debaters.map((d) => (
                <button
                  key={d.id}
                  className={`vote ${myVote === d.id ? "on" : ""}`}
                  style={{ borderColor: myVote === d.id ? d.color_hex : undefined }}
                  disabled={phase === "idle"}
                  onClick={() => vote(d.id)}
                >
                  <span className="dot" style={{ background: d.color_hex }} />
                  {d.name}
                </button>
              ))}
              {debaters.length === 0 && <span className="muted">No debaters yet.</span>}
            </div>
            <div className="react-row">
              {REACTIONS.map((r) => (
                <button
                  key={r.key}
                  disabled={phase === "idle"}
                  onClick={() => act(() => api.stageReact(runId, r.key, speaking?.kind === "debate" ? speaking.speaker_id : null))}
                >
                  {r.label}
                </button>
              ))}
            </div>
            <NarrateForm runId={runId} onError={setError} />
            {mode === "stage" && (
              <p className="faint audience-link">
                Audience page: <a href={audienceUrl}>{audienceUrl}</a>
              </p>
            )}
          </div>
        </section>
      )}
    </div>
  );
}

interface PodiumProps {
  debater: Debater;
  line: StageLine | null;
  last: StageLine | undefined;
  votes: number;
  total: number;
  winner: boolean;
  bursts: { id: number; reaction: StageReaction }[];
  side: "left" | "right";
}

function Podium({ debater, line, last, votes, total, winner, bursts, side }: PodiumProps) {
  const caption = line ?? last;
  return (
    <div
      className={`podium podium-${side} ${line ? "speaking" : ""} ${winner ? "winner" : ""}`}
      style={{ ["--c" as string]: debater.color_hex }}
    >
      <div className="avatar">
        <span>{initials(debater.name)}</span>
        {bursts.map((b) => (
          <span key={b.id} className={`burst burst-${b.reaction}`}>
            {REACTION_MARK[b.reaction]}
          </span>
        ))}
      </div>
      <div className="podium-name">{debater.name}</div>
      <div className="podium-country">
        {debater.country} · <span className="mono">{debater.voice}</span>
      </div>
      <div className={`caption ${line ? "live" : ""}`}>
        {caption ? (
          <>
            {caption.emotion && <span className="emotion">{caption.emotion}</span>}
            {caption.text}
          </>
        ) : (
          <span className="faint">…</span>
        )}
      </div>
      <div className="votebar" title={`${votes} vote${votes === 1 ? "" : "s"}`}>
        <span style={{ width: total ? `${(votes / total) * 100}%` : 0 }} />
      </div>
      <div className="mono faint">
        {votes} vote{votes === 1 ? "" : "s"}
        {winner ? " · winner" : ""}
      </div>
    </div>
  );
}

function NarrateForm({ runId, onError }: { runId: string; onError: (e: string | null) => void }) {
  const [text, setText] = useState("");
  const [name, setName] = useState(() => localStorage.getItem("wwp-narrator-name") ?? "");
  const [announce, setAnnounce] = useState(false);
  const [pending, setPending] = useState(false);
  const [listening, setListening] = useState(false);
  const Recognition =
    (window as unknown as { SpeechRecognition?: new () => SpeechRecognitionLike }).SpeechRecognition ??
    (window as unknown as { webkitSpeechRecognition?: new () => SpeechRecognitionLike }).webkitSpeechRecognition;

  const submit = async (e?: FormEvent, override?: string) => {
    e?.preventDefault();
    const t = (override ?? text).trim();
    if (!t || pending) return;
    setPending(true);
    onError(null);
    try {
      localStorage.setItem("wwp-narrator-name", name);
      await api.stageNarrate(runId, t, name.trim() || "Narrator", announce);
      setText("");
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setPending(false);
    }
  };

  const listen = () => {
    if (!Recognition || listening) return;
    const rec = new Recognition();
    rec.lang = "en-US";
    rec.interimResults = false;
    rec.onresult = (ev) => {
      const said = ev.results[0]?.[0]?.transcript ?? "";
      setText(said);
      void submit(undefined, said);
    };
    rec.onend = () => setListening(false);
    setListening(true);
    rec.start();
  };

  return (
    <form className="narrate" onSubmit={submit}>
      <div className="narrate-row">
        <input
          className="narrate-name"
          value={name}
          maxLength={40}
          placeholder="Your name"
          onChange={(e) => setName(e.target.value)}
        />
        <input
          className="narrate-text"
          value={text}
          maxLength={280}
          placeholder="Narrate: say something the debaters will hear"
          onChange={(e) => setText(e.target.value)}
        />
      </div>
      <div className="narrate-row">
        <label className="check">
          <input type="checkbox" checked={announce} onChange={(e) => setAnnounce(e.target.checked)} />
          Announce to every agent
        </label>
        <span className="spacer" />
        {Recognition && (
          <button type="button" className={listening ? "on" : ""} onClick={listen} title="Speak your line">
            {listening ? "Listening…" : "Mic"}
          </button>
        )}
        <button className="primary" disabled={!text.trim() || pending}>
          {pending ? "Sending…" : "Narrate"}
        </button>
      </div>
    </form>
  );
}

interface SpeechRecognitionLike {
  lang: string;
  interimResults: boolean;
  onresult: ((ev: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null;
  onend: (() => void) | null;
  start(): void;
}

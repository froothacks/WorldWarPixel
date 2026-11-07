import { useEffect, useRef, useState, type FormEvent } from "react";
import { api } from "./api";
import CanvasView from "./CanvasView";
import { useDebate, useDebateFocus } from "./debate";
import type { StageReaction } from "./api";

const REACTIONS: StageReaction[] = ["cheer", "boo", "laugh", "gasp"];

interface SpeechResultEvent {
  results: ArrayLike<ArrayLike<{ transcript: string }>>;
}

interface SpeechRecognitionLike {
  lang: string;
  onresult: ((event: SpeechResultEvent) => void) | null;
  onend: (() => void) | null;
  onerror: (() => void) | null;
  start: () => void;
  stop: () => void;
}

interface SpeechRecognitionWindow extends Window {
  SpeechRecognition?: new () => SpeechRecognitionLike;
  webkitSpeechRecognition?: new () => SpeechRecognitionLike;
}

function getVoterId() {
  let id = localStorage.getItem("wwp-voter");
  if (!id) {
    id = Math.random().toString(36).slice(2, 12);
    localStorage.setItem("wwp-voter", id);
  }
  return id;
}

export default function Audience({ runId }: { runId: string }) {
  const debate = useDebate(runId);
  const state = debate.state;
  const focus = useDebateFocus(state);
  const line = debate.transcript.at(-1) ?? null;
  const [name, setName] = useState(
    () => localStorage.getItem("wwp-narrator-name") ?? "Host",
  );
  const [text, setText] = useState("");
  const [myVote, setMyVote] = useState<number | null>(() => {
    const saved = localStorage.getItem(`wwp-vote-${runId}`);
    return saved ? Number(saved) : null;
  });
  const [listening, setListening] = useState(false);
  const recognition = useRef<SpeechRecognitionLike | null>(null);
  const active = Boolean(state && state.phase !== "idle");

  useEffect(() => {
    document.documentElement.classList.add("stage-page");
    return () => {
      document.documentElement.classList.remove("stage-page");
      recognition.current?.stop();
    };
  }, []);

  const vote = async (agentId: number) => {
    try {
      await api.stageVote(runId, getVoterId(), agentId);
      setMyVote(agentId);
      localStorage.setItem(`wwp-vote-${runId}`, String(agentId));
    } catch (e) {
      alert((e as Error).message);
    }
  };
  const react = async (reaction: StageReaction) => {
    try {
      await api.stageReact(runId, reaction, null);
    } catch (e) {
      alert((e as Error).message);
    }
  };
  const say = async (event: FormEvent) => {
    event.preventDefault();
    const message = text.trim();
    if (!message) return;
    const speaker = name.trim() || "Host";
    localStorage.setItem("wwp-narrator-name", speaker);
    try {
      await api.stageNarrate(runId, message, speaker);
      setText("");
    } catch (e) {
      alert((e as Error).message);
    }
  };
  const startMic = () => {
    const SpeechRecognition =
      (window as SpeechRecognitionWindow).SpeechRecognition ??
      (window as SpeechRecognitionWindow).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Speech recognition is not available in this browser.");
      return;
    }
    const next = new SpeechRecognition();
    next.lang = "en-US";
    next.onresult = (event) => {
      const result = event.results[0]?.[0]?.transcript;
      if (result) setText(result);
    };
    next.onend = () => setListening(false);
    next.onerror = () => setListening(false);
    recognition.current = next;
    setListening(true);
    next.start();
  };
  const debaters = state?.debaters ?? [];

  return (
    <main className="audience">
      <header className="audience-header">
        <strong>WorldWarPixel</strong>
        <span className="audience-phase">{state?.phase ?? "connecting"}</span>
        <span className="audience-debate">
          {state?.phase === "idle"
            ? "Waiting for the next debate"
            : `Debate #${state?.debate_id} · ${state?.topic}`}
        </span>
      </header>
      <section className="audience-canvas">
        <CanvasView
          store={debate.store}
          version={debate.version}
          agents={debate.agentMap}
          highlightAgent={null}
          highlightMask={null}
          onPickAgent={() => undefined}
          focus={focus}
          focusScale={1}
        />
      </section>
      {line && (
        <section className="audience-caption">
          <div className="audience-caption-label">
            {line.kind === "verdict" ? "Verdict" : line.name}
            {line.country && ` · ${line.country}`}
            {line.kind === "debate" && line.emotion && <span>{line.emotion}</span>}
          </div>
          <p>{line.text}</p>
        </section>
      )}
      <section className="audience-votes">
        {debaters.map((debater) => (
          <button
            key={debater.id}
            className={`audience-vote ${myVote === debater.id ? "selected" : ""}`}
            style={
              myVote === debater.id
                ? { borderColor: debater.color_hex, boxShadow: `0 0 0 1px ${debater.color_hex}` }
                : undefined
            }
            disabled={!active}
            onClick={() => void vote(debater.id)}
          >
            <span className="debate-swatch" style={{ background: debater.color_hex }} />
            <span>{debater.name}</span>
            <strong>{state?.votes[String(debater.id)] ?? 0}</strong>
          </button>
        ))}
      </section>
      <section className="audience-reactions">
        {REACTIONS.map((reaction) => (
          <button key={reaction} disabled={!active} onClick={() => void react(reaction)}>
            {reaction}
          </button>
        ))}
      </section>
      <form className="audience-say" onSubmit={say}>
        <label htmlFor="audience-name">Say something on stage</label>
        <input
          id="audience-name"
          value={name}
          maxLength={40}
          placeholder="Host"
          onChange={(event) => setName(event.target.value)}
        />
        <textarea
          value={text}
          maxLength={280}
          placeholder="Say something on stage. The debaters will hear it."
          onChange={(event) => setText(event.target.value)}
        />
        <div className="audience-say-actions">
          <button type="button" onClick={startMic} disabled={listening}>
            {listening ? "Listening…" : "Mic"}
          </button>
          <button type="submit" className="primary" disabled={!text.trim()}>
            Say
          </button>
        </div>
      </form>
    </main>
  );
}

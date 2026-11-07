import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import type { NarratorHighlight, NarratorState, NarratorUpdate, StageLine } from "./api";

const EVENT_MARK: Record<string, string> = {
  finished: "+",
  team: "&",
  raid: "x",
  repair: "+",
  overwrite: "~",
  started: "*",
  move: ">",
  thread: "#",
  hot_thread: "!",
};

interface Props {
  narrator: NarratorState | null;
  paused: boolean;
  stageLines: StageLine[];
  onHighlight: (highlight: NarratorHighlight) => void;
  onAsk: (question: string) => Promise<void>;
  onSay: (text: string, name: string) => Promise<void>;
}

const CHIP_PREFIX = { agent: "@", thread: "#", region: "" } as const;
const EMPTY_UPDATES: NarratorUpdate[] = [];

export default function NarratorDock({
  narrator,
  paused,
  stageLines,
  onHighlight,
  onAsk,
  onSay,
}: Props) {
  const [question, setQuestion] = useState("");
  const [sayText, setSayText] = useState("");
  const [hostName, setHostName] = useState(
    () => localStorage.getItem("wwp-narrator-name") ?? "Host",
  );
  const [mode, setMode] = useState<"ask" | "say">("ask");
  const [pending, setPending] = useState(false);
  const [asked, setAsked] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now() / 1000);
  const [showLog, setShowLog] = useState(false);
  const [expanded, setExpanded] = useState<Set<number>>(new Set());
  const logRef = useRef<HTMLDivElement>(null);
  const followRef = useRef(true);
  const updates = narrator?.updates ?? EMPTY_UPDATES;
  const entries = useMemo(
    () =>
      [
        ...updates.map((update) => ({ type: "update" as const, ts: update.ts, update })),
        ...stageLines.map((line) => ({ type: "stage" as const, ts: line.ts, line })),
      ].sort((a, b) => a.ts - b.ts),
    [stageLines, updates],
  );
  const count = entries.length;
  const latest = updates.at(-1);
  const newest = entries.at(-1);
  const hasModel = narrator?.model ?? true;

  // Newest entry at the bottom of the log: follow it unless the viewer scrolled up.
  useEffect(() => {
    const el = logRef.current;
    if (el && followRef.current) el.scrollTop = el.scrollHeight;
  }, [count, pending, showLog]);

  const onScroll = () => {
    const el = logRef.current;
    if (el) followRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  };

  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => window.clearInterval(t);
  }, []);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const q = question.trim();
    if (!q || pending) return;
    setPending(true);
    setAsked(q);
    setError(null);
    try {
      await onAsk(q);
      setQuestion("");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setPending(false);
    }
  };

  const submitSay = async (e: FormEvent) => {
    e.preventDefault();
    const text = sayText.trim();
    if (!text || pending) return;
    const name = hostName.trim() || "Host";
    setPending(true);
    setAsked(text);
    setError(null);
    localStorage.setItem("wwp-narrator-name", name);
    try {
      await onSay(text, name);
      setSayText("");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setPending(false);
    }
  };

  const toggle = (id: number) =>
    setExpanded((s) => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const interval = narrator?.interval ?? 60;
  const lastUpdate = [...updates].reverse().find((u) => u.kind === "update");
  const nextAt = lastUpdate ? lastUpdate.ts + interval : (narrator?.next_update_at ?? null);
  const countdown = nextAt === null ? null : Math.max(0, Math.round(nextAt - now));
  const failures = narrator?.stats.failures ?? 0;
  // A fresh answer opens by default; everything else is one line until clicked.
  const isOpen = (u: NarratorUpdate) => expanded.has(u.id) !== (u.kind === "answer" && u.id === latest?.id);

  return (
    <footer className="narrator">
      <div className="narrator-head">
        <span className="label">Narrator</span>
        <span className="meta">
          every {interval}s{paused ? " · paused" : countdown !== null ? ` · next ${countdown}s` : ""}
          {failures > 0 && ` · ${failures} failed`}
        </span>
        <span className="spacer" />
        {count > 1 && (
          <button type="button" className="text" onClick={() => setShowLog(!showLog)}>
            {showLog ? "hide log" : `log (${count})`}
          </button>
        )}
      </div>
      {showLog ? (
        <div className="narrator-log" ref={logRef} onScroll={onScroll}>
          {entries.map((entry) => {
            if (entry.type === "update") {
              return (
                <UpdateItem
                  key={`update-${entry.update.id}`}
                  update={entry.update}
                  open={isOpen(entry.update)}
                  onToggle={() => toggle(entry.update.id)}
                  onHighlight={onHighlight}
                />
              );
            }
            return <StageLineItem key={`stage-${entry.line.id}`} line={entry.line} />;
          })}
        </div>
      ) : newest ? (
        newest.type === "update" ? (
          <UpdateItem update={newest.update} open={isOpen(newest.update)} onToggle={() => toggle(newest.update.id)} onHighlight={onHighlight} />
        ) : (
          <StageLineItem line={newest.line} />
        )
      ) : (
        <p className="narrator-empty">{narrator ? `First update in ~${interval}s. You can ask a question now.` : "Connecting…"}</p>
      )}
      {pending && (
        <div className="narr-pending">
          {asked} <span className="faint">· thinking…</span>
        </div>
      )}
      <div className="narrator-mode">
        <button className={mode === "ask" ? "on" : ""} onClick={() => setMode("ask")}>Ask</button>
        <button className={mode === "say" ? "on" : ""} onClick={() => setMode("say")}>Say</button>
      </div>
      {mode === "ask" ? (
        <form className="narrator-ask" onSubmit={submit}>
          <input
            placeholder={hasModel ? "Ask the narrator, e.g. who holds the top-left corner?" : "No narrator model for this run (random agents)"}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={pending || !narrator || !hasModel}
            maxLength={500}
          />
          {error && <span className="error" title={error}>{error}</span>}
          <button type="submit" disabled={pending || !narrator || !hasModel || !question.trim()}>
            {pending ? "asking…" : "ask"}
          </button>
        </form>
      ) : (
        <form className="narrator-ask narrator-say" onSubmit={submitSay}>
          <input
            className="narrator-name"
            placeholder="Host"
            value={hostName}
            onChange={(e) => setHostName(e.target.value)}
            maxLength={40}
          />
          <input
            placeholder="Say something on stage. The debaters will hear it."
            value={sayText}
            onChange={(e) => setSayText(e.target.value)}
            disabled={pending}
            maxLength={280}
          />
          {error && <span className="error" title={error}>{error}</span>}
          <button type="submit" disabled={pending || !sayText.trim()}>
            {pending ? "sending…" : "say"}
          </button>
        </form>
      )}
    </footer>
  );
}

function StageLineItem({ line }: { line: StageLine }) {
  const time = new Date(line.ts * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
  const label = line.kind === "verdict" ? "Verdict" : line.name;
  return (
    <div className={`narr-stage-line stage-line-${line.kind}`}>
      <span className="narr-time">{time}</span>
      {line.kind === "debate" && (
        <span className="narr-stage-swatch" style={{ background: line.color_hex ?? "var(--accent)" }} />
      )}
      <strong>{label}</strong>
      {line.kind === "debate" && line.emotion && <span className="faint">({line.emotion})</span>}
      <span className="narr-stage-text">{line.text}</span>
    </div>
  );
}

interface ItemProps {
  update: NarratorUpdate;
  open: boolean;
  onToggle: () => void;
  onHighlight: Props["onHighlight"];
}

function UpdateItem({ update: u, open, onToggle, onHighlight }: ItemProps) {
  const time = new Date(u.ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
  const events = u.kind === "update" ? (u.events ?? []) : [];
  const hasDetail = u.bullets.length > 0 || events.length > 0 || u.question !== null || u.highlights.length > 0;
  return (
    <div className={`narr-item ${u.kind} ${open ? "open" : ""}`}>
      <div className="narr-row">
        <span className="narr-time">{time}</span>
        {u.kind === "answer" && <span className="narr-kind">Answer</span>}
        <button type="button" className="narr-headline" onClick={onToggle} disabled={!hasDetail} title={open ? "Collapse" : "Expand"}>
          {u.headline}
        </button>
        {!open && u.highlights.length > 0 && <Chips highlights={u.highlights} onHighlight={onHighlight} />}
        {hasDetail && <span className="narr-toggle">{open ? "−" : "+"}</span>}
      </div>
      {open && (
        <div className="narr-detail">
          {u.question && <div className="narr-question">{u.question}</div>}
          {u.highlights.length > 0 && <Chips highlights={u.highlights} onHighlight={onHighlight} />}
          {u.bullets.length > 0 && (
            <ul className="narr-bullets">
              {u.bullets.map((b, i) => (
                <li key={i}>{b}</li>
              ))}
            </ul>
          )}
          {events.length > 0 && (
            <details className="narr-events">
              <summary>{events.length} events</summary>
              <ul>
                {events.map((e, i) => (
                  <li key={i} className={`ev ev-${e.kind}`}>
                    <span className="ev-kind">{EVENT_MARK[e.kind] ?? "·"}</span>
                    {e.text}
                  </li>
                ))}
              </ul>
            </details>
          )}
          <span className="narr-note">
            {u.kind === "update" && `last ${Math.round(u.window.seconds)}s: ${u.window.turns} turns, ${u.window.pixels} px, ${u.window.posts} posts`}
            {u.source === "fallback" && (u.error ? " · stats only, model unavailable" : " · stats only")}
          </span>
        </div>
      )}
    </div>
  );
}

function Chips({ highlights, onHighlight }: { highlights: NarratorHighlight[]; onHighlight: Props["onHighlight"] }) {
  return (
    <span className="chips">
      {highlights.map((h, i) => (
        <button
          key={i}
          type="button"
          className={`chip chip-${h.type}`}
          title={h.type === "region" ? `Pan the canvas to (${h.x}, ${h.y}) ${h.w}×${h.h}` : h.type === "agent" ? "Select this agent" : "Open this thread"}
          onClick={() => onHighlight(h)}
        >
          {CHIP_PREFIX[h.type]}
          {h.label}
        </button>
      ))}
    </span>
  );
}

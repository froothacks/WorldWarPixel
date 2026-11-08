import { useEffect, useRef, useState, type FormEvent } from "react";
import { InlineError } from "./Errors";
import type { NarratorHighlight, NarratorState, NarratorUpdate } from "./api";

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
  onHighlight: (highlight: NarratorHighlight) => void;
  onAsk: (question: string) => Promise<void>;
}

const CHIP_PREFIX = { agent: "@", thread: "#", region: "" } as const;

export default function NarratorDock({ narrator, paused, onHighlight, onAsk }: Props) {
  const [question, setQuestion] = useState("");
  const [pending, setPending] = useState(false);
  const [asked, setAsked] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now() / 1000);
  const [showLog, setShowLog] = useState(false);
  const [expanded, setExpanded] = useState<Set<number>>(new Set());
  const logRef = useRef<HTMLDivElement>(null);
  const followRef = useRef(true);
  const updates = narrator?.updates ?? [];
  const count = updates.length;
  const latest = updates.at(-1);
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
            {showLog ? "Hide history" : `History (${count})`}
          </button>
        )}
      </div>
      {showLog ? (
        <div className="narrator-log" ref={logRef} onScroll={onScroll}>
          {updates.map((u) => (
            <UpdateItem key={u.id} update={u} open={isOpen(u)} onToggle={() => toggle(u.id)} onHighlight={onHighlight} />
          ))}
        </div>
      ) : latest ? (
        <UpdateItem update={latest} open={isOpen(latest)} onToggle={() => toggle(latest.id)} onHighlight={onHighlight} />
      ) : (
        <p className="narrator-empty">{narrator ? `First update in ~${interval}s. You can ask a question now.` : "Connecting…"}</p>
      )}
      {pending && (
        <div className="narr-pending">
          {asked} <span className="faint">· thinking…</span>
        </div>
      )}
      <form className="narrator-ask" onSubmit={submit}>
        <input
          placeholder={hasModel ? "Ask the narrator, e.g. who holds the top-left corner?" : "No narrator model for this run (random agents)"}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={pending || !narrator || !hasModel}
          maxLength={500}
        />
        {error && <InlineError message={error} />}
        <button type="submit" disabled={pending || !narrator || !hasModel || !question.trim()}>
          {pending ? "Asking…" : "Ask"}
        </button>
      </form>
    </footer>
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
      <span className="narr-time">{time}</span>
      <div className="narr-body">
        <div className="narr-row">
          {u.kind === "answer" && <span className="narr-kind">Answer</span>}
          <button type="button" className="narr-headline" onClick={onToggle} disabled={!hasDetail} title={open ? "Collapse" : "Expand"}>
            {u.headline}
          </button>
          {hasDetail && <span className="narr-toggle">{open ? "−" : "+"}</span>}
        </div>
        {!open && u.highlights.length > 0 && <Chips highlights={u.highlights} onHighlight={onHighlight} />}
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

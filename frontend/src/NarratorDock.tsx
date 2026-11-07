import { useEffect, useRef, useState, type FormEvent } from "react";
import type { NarratorHighlight, NarratorState, NarratorUpdate, RunStats } from "./api";

interface Props {
  narrator: NarratorState | null;
  stats: RunStats;
  threads: number;
  onHighlight: (highlight: NarratorHighlight) => void;
  onAsk: (question: string) => Promise<void>;
}

const CHIP_ICON = { agent: "@", thread: "#", region: "⌖" } as const;

export default function NarratorDock({ narrator, stats, threads, onHighlight, onAsk }: Props) {
  const [question, setQuestion] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now() / 1000);
  const bodyRef = useRef<HTMLDivElement>(null);
  const updates = narrator?.updates ?? [];
  const count = updates.length;

  // Newest update at the bottom, like a chat: follow it as it arrives.
  useEffect(() => {
    const el = bodyRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [count, pending]);

  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => window.clearInterval(t);
  }, []);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const q = question.trim();
    if (!q || pending) return;
    setPending(true);
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

  const interval = narrator?.interval ?? 60;
  const lastUpdate = [...updates].reverse().find((u) => u.kind === "update");
  const nextAt = lastUpdate ? lastUpdate.ts + interval : narrator?.next_update_at ?? null;
  const countdown = nextAt === null ? null : Math.max(0, Math.round(nextAt - now));

  return (
    <footer className="narrator">
      <div className="narrator-head">
        <strong>Narrator</strong>
        <span className="muted small">
          every {interval} s{countdown !== null && ` · next in ${countdown} s`}
          {narrator && narrator.stats.failures > 0 && ` · ${narrator.stats.failures} failed model calls`}
        </span>
        <span className="spacer" />
        <span className="muted small">
          {stats.turns.toLocaleString()} turns · {stats.pixels.toLocaleString()} px · {stats.posts.toLocaleString()} posts · {threads} threads
        </span>
      </div>
      <div className="narrator-body" ref={bodyRef}>
        {count === 0 && (
          <p className="narrator-empty small">
            {narrator ? `Waiting for the first update (every ${interval} s). Ask a question below in the meantime.` : "Connecting…"}
          </p>
        )}
        {updates.map((u) => (
          <UpdateItem key={u.id} update={u} onHighlight={onHighlight} />
        ))}
        {pending && <p className="muted small">Narrator is thinking…</p>}
      </div>
      <form className="narrator-ask" onSubmit={submit}>
        <input
          placeholder="Ask the narrator… e.g. who is winning the top-left corner?"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={pending || !narrator}
          maxLength={500}
        />
        <button type="submit" disabled={pending || !narrator || !question.trim()}>
          {pending ? "Asking…" : "Ask"}
        </button>
        {error && <span className="error" title={error}>{error}</span>}
      </form>
    </footer>
  );
}

function UpdateItem({ update: u, onHighlight }: { update: NarratorUpdate; onHighlight: Props["onHighlight"] }) {
  const time = new Date(u.ts * 1000).toLocaleTimeString();
  return (
    <>
      {u.kind === "answer" && u.question && <div className="narr-question">{u.question}</div>}
      <div className={`narr-item ${u.kind}`}>
        <div className="narr-meta">
          <span className="kind">{u.kind}</span>
          <span>{time}</span>
          {u.kind === "update" && (
            <span>
              · last {Math.round(u.window.seconds)} s: {u.window.turns} turns, {u.window.pixels} px, {u.window.posts} posts
            </span>
          )}
          {u.source === "fallback" && <span title={u.error ?? undefined}>· stats only, model unavailable</span>}
        </div>
        <div className="narr-headline">{u.headline}</div>
        {u.bullets.length > 0 && (
          <ul className="narr-bullets">
            {u.bullets.map((b, i) => (
              <li key={i}>{b}</li>
            ))}
          </ul>
        )}
        {u.highlights.length > 0 && (
          <div className="chips">
            {u.highlights.map((h, i) => (
              <button
                key={i}
                type="button"
                className={`chip chip-${h.type}`}
                title={h.type === "region" ? `Pan the canvas to (${h.x}, ${h.y}) ${h.w}×${h.h}` : h.type === "agent" ? "Select this agent" : "Open this thread"}
                onClick={() => onHighlight(h)}
              >
                {CHIP_ICON[h.type]} {h.label}
              </button>
            ))}
          </div>
        )}
      </div>
    </>
  );
}

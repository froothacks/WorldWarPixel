import { useEffect, useState } from "react";
import { api, type Post, type ThreadDigest } from "./api";

interface Props {
  runId: string;
  threads: ThreadDigest[];
  agentName: (id: number) => string;
  version: number;
  totalThreads: number;
}

export default function Board({ runId, threads, agentName, version, totalThreads }: Props) {
  const [open, setOpen] = useState<number | null>(null);
  const [posts, setPosts] = useState<Post[]>([]);

  useEffect(() => {
    if (open === null) return;
    api.thread(runId, open).then(setPosts).catch(() => setPosts([]));
  }, [runId, open, version]);

  return (
    <aside className="board">
      <header className="board-header">
        <strong>Message board</strong>
        <span className="muted small">{totalThreads} threads{totalThreads > threads.length ? ` · showing ${threads.length} most active` : ""}</span>
      </header>
      <div className="threads">
        {threads.length === 0 && <p className="muted small pad">No posts yet.</p>}
        {threads.map((t) => (
          <div key={t.thread_id} className={`thread ${open === t.thread_id ? "open" : ""}`}>
            <button className="thread-title" onClick={() => setOpen(open === t.thread_id ? null : t.thread_id)}>
              <span>{t.title}</span>
              <span className="muted small">{t.author} · {t.replies} {t.replies === 1 ? "reply" : "replies"}</span>
            </button>
            {open === t.thread_id ? (
              <div className="replies">
                {posts.map((p) => (
                  <div key={p.id} className={`post ${p.parent_id ? "nested" : ""}`}>
                    <span className="author">{agentName(p.author_id)}</span> {p.body}
                  </div>
                ))}
              </div>
            ) : (
              t.latest && (
                <div className="latest small">
                  <span className="author">{t.latest.author}</span> {t.latest.body}
                </div>
              )
            )}
          </div>
        ))}
      </div>
    </aside>
  );
}

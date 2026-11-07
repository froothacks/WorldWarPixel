import { useEffect, useRef, useState } from "react";
import { api, type Post, type ThreadDigest } from "./api";

interface Props {
  runId: string;
  threads: ThreadDigest[];
  agentName: (id: number) => string;
  version: number;
  totalThreads: number;
  open: number | null;
  onOpen: (threadId: number | null) => void;
  onVoted: () => void;
}

type MyVotes = Record<number, 1 | -1>;

function loadVotes(runId: string): MyVotes {
  try {
    return JSON.parse(localStorage.getItem(`wwp-votes-${runId}`) ?? "{}") as MyVotes;
  } catch {
    return {};
  }
}

function voterId(): string {
  let id = localStorage.getItem("wwp-voter");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("wwp-voter", id);
  }
  return id;
}

export default function Board({ runId, threads, agentName, version, totalThreads, open, onOpen, onVoted }: Props) {
  const [posts, setPosts] = useState<Post[]>([]);
  const [myVotes, setMyVotes] = useState<MyVotes>(() => loadVotes(runId));

  const vote = (threadId: number, dir: 1 | -1) => {
    const prev = myVotes[threadId] ?? 0;
    const next = prev === dir ? 0 : dir;
    const updated = { ...myVotes };
    if (next === 0) delete updated[threadId];
    else updated[threadId] = next;
    const save = (v: MyVotes) => {
      setMyVotes(v);
      localStorage.setItem(`wwp-votes-${runId}`, JSON.stringify(v));
    };
    save(updated);
    api
      .vote(runId, threadId, voterId(), next)
      .then(onVoted)
      .catch(() => save(myVotes));
  };

  const voter = (threadId: number, votes: number) => (
    <div className="voter">
      <button
        className={`vote ${myVotes[threadId] === 1 ? "on" : ""}`}
        title="Upvote: move this thread up the board (agents see the board in this order)"
        onClick={() => vote(threadId, 1)}
      >
        ▲
      </button>
      <span className="small">{votes}</span>
      <button
        className={`vote ${myVotes[threadId] === -1 ? "on" : ""}`}
        title="Downvote: move this thread down the board"
        onClick={() => vote(threadId, -1)}
      >
        ▼
      </button>
    </div>
  );
  const setOpen = onOpen;
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (open === null) return;
    api.thread(runId, open).then(setPosts).catch(() => setPosts([]));
  }, [runId, open, version]);

  useEffect(() => {
    if (open === null) return;
    listRef.current?.querySelector(".thread.open")?.scrollIntoView({ block: "nearest" });
  }, [open, posts]);

  return (
    <aside className="board">
      <header className="board-header">
        <strong>Message board</strong>
        <span className="muted small" title="Sorted by most recent activity; each net vote moves a thread up or down">
          {totalThreads} threads · recent activity ± votes{totalThreads > threads.length ? ` · top ${threads.length}` : ""}
        </span>
      </header>
      <div className="threads" ref={listRef}>
        {threads.length === 0 && <p className="muted small pad">No posts yet.</p>}
        {open !== null && posts[0] && !threads.some((t) => t.thread_id === open) && (
          <div className="thread open">
            <button className="thread-title" onClick={() => setOpen(null)}>
              <span>{posts[0].title}</span>
              <span className="muted small">{agentName(posts[0].author_id)} · {posts.length - 1} {posts.length === 2 ? "reply" : "replies"}</span>
            </button>
            <div className="replies">
              {posts.map((p) => (
                <div key={p.id} className={`post ${p.parent_id ? "nested" : ""}`}>
                  <span className="author">{agentName(p.author_id)}</span> {p.body}
                </div>
              ))}
            </div>
          </div>
        )}
        {threads.map((t) => (
          <div key={t.thread_id} className={`thread ${open === t.thread_id ? "open" : ""}`}>
            <div className="thread-row">
              {voter(t.thread_id, t.votes ?? 0)}
              <button className="thread-title" onClick={() => setOpen(open === t.thread_id ? null : t.thread_id)}>
                <span>{t.title}</span>
                <span className="muted small">{t.author} · {t.replies} {t.replies === 1 ? "reply" : "replies"}</span>
              </button>
            </div>
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

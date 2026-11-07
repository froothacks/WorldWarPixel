import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { api, GOD_ID, type Post, type ThreadDigest } from "./api";

interface Props {
  runId: string;
  threads: ThreadDigest[];
  agentName: (id: number) => string;
  agentColor: (id: number) => string;
  onSelectAgent: (id: number) => void;
  version: number;
  totalThreads: number;
  open: number | null;
  onOpen: (threadId: number | null) => void;
  onVoted: () => void;
  hidden?: boolean;
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
    id =
      typeof crypto.randomUUID === "function"
        ? crypto.randomUUID()
        : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
    localStorage.setItem("wwp-voter", id);
  }
  return id;
}

/** Seconds of recency one net vote is worth; mirrors VOTE_BOOST in sim/board.py. */
const VOTE_BOOST = 60;
const rank = (t: ThreadDigest) => t.last_activity + (t.votes ?? 0) * VOTE_BOOST;

type Sort = "active" | "new" | "top";

const SORTS: { key: Sort; label: string; title: string }[] = [
  { key: "active", label: "Active", title: "Most recent activity first; each net vote moves a thread up or down" },
  { key: "new", label: "New", title: "Newest threads first" },
  { key: "top", label: "Top", title: "Most replies first" },
];

const sorters: Record<Sort, (a: ThreadDigest, b: ThreadDigest) => number> = {
  active: (a, b) => rank(b) - rank(a),
  new: (a, b) => b.created_at - a.created_at,
  top: (a, b) => b.replies - a.replies || b.last_activity - a.last_activity,
};

export default function Board({ runId, threads, agentName, agentColor, onSelectAgent, version, totalThreads, open, onOpen, onVoted, hidden = false }: Props) {
  const now = useNow();
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<Sort>("active");
  const [seen, setSeen] = useState<Map<number, number>>(() => new Map());
  const [mountedAt] = useState(() => Date.now() / 1000);
  const [frozen, setFrozen] = useState<number[] | null>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const listScroll = useRef(0);
  const [myVotes, setMyVotes] = useState<MyVotes>(() => loadVotes(runId));
  const voteQueue = useRef<Promise<unknown>>(Promise.resolve());

  const saveVote = (threadId: number, value: 0 | 1 | -1) => {
    const v = loadVotes(runId);
    if (value === 0) delete v[threadId];
    else v[threadId] = value;
    localStorage.setItem(`wwp-votes-${runId}`, JSON.stringify(v));
    setMyVotes(v);
  };

  const vote = (threadId: number, dir: 1 | -1) => {
    const prev = loadVotes(runId)[threadId] ?? 0;
    const next = prev === dir ? 0 : dir;
    saveVote(threadId, next);
    // Send votes one at a time so the server ends up with the last click.
    voteQueue.current = voteQueue.current.then(() =>
      api
        .vote(runId, threadId, voterId(), next)
        .then(onVoted)
        .catch(() => {
          if ((loadVotes(runId)[threadId] ?? 0) === next) saveVote(threadId, prev);
        }),
    );
  };

  const isUnread = useCallback(
    (t: ThreadDigest) => t.last_activity > (seen.get(t.thread_id) ?? mountedAt),
    [seen, mountedAt],
  );

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    const matches = q
      ? threads.filter((t) =>
          [t.title, t.author, t.latest?.author ?? "", t.latest?.body ?? ""].some((s) => s.toLowerCase().includes(q)),
        )
      : threads;
    const sorted = [...matches].sort(sorters[sort]);
    if (!frozen) return sorted;
    // Don't reshuffle rows under the pointer; new threads wait at the bottom until it leaves.
    const rank = new Map(frozen.map((id, i) => [id, i]));
    return sorted.sort((a, b) => (rank.get(a.thread_id) ?? Infinity) - (rank.get(b.thread_id) ?? Infinity));
  }, [threads, query, sort, frozen]);

  const unread = threads.filter(isUnread).length;
  const openDigest = threads.find((t) => t.thread_id === open);

  const markSeen = useCallback((id: number, ts: number) => {
    setSeen((m) => ((m.get(id) ?? 0) >= ts ? m : new Map(m).set(id, ts)));
  }, []);

  const openThread = (id: number) => {
    listScroll.current = listRef.current?.scrollTop ?? 0;
    onOpen(id);
  };

  useLayoutEffect(() => {
    if (open === null && listRef.current) listRef.current.scrollTop = listScroll.current;
  }, [open]);

  const onListKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    const rows = Array.from(listRef.current?.querySelectorAll<HTMLButtonElement>(".thread-row") ?? []);
    if (rows.length === 0) return;
    e.preventDefault();
    const current = document.activeElement?.closest(".thread-item")?.querySelector(".thread-row");
    const i = rows.indexOf(current as HTMLButtonElement);
    const next = e.key === "ArrowDown" ? Math.min(rows.length - 1, i + 1) : Math.max(0, i - 1);
    rows[next].focus();
  };

  if (open !== null) {
    return (
      <aside className="board" hidden={hidden}>
        <ThreadView
          key={open}
          runId={runId}
          threadId={open}
          digest={openDigest}
          version={version}
          now={now}
          agentName={agentName}
          agentColor={agentColor}
          onSelectAgent={onSelectAgent}
          onSeen={markSeen}
          onBack={() => onOpen(null)}
          active={!hidden}
        />
      </aside>
    );
  }

  return (
    <aside className="board" hidden={hidden}>
      <header className="board-header">
        <strong>Message board</strong>
        <span className="muted small">
          {totalThreads} {totalThreads === 1 ? "thread" : "threads"}
          {unread > 0 && <span className="unread-count"> · {unread} new</span>}
        </span>
        {unread > 0 && (
          <button
            className="ghost small mark-read"
            onClick={() => setSeen(new Map(threads.map((t) => [t.thread_id, t.last_activity])))}
            title="Mark every thread as read"
          >
            Mark read
          </button>
        )}
      </header>
      <div className="board-tools">
        <input
          className="board-search"
          type="search"
          placeholder="Search threads, authors, replies"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Escape" && setQuery("")}
        />
        <div className="segmented" role="tablist" aria-label="Sort threads">
          {SORTS.map((s) => (
            <button
              key={s.key}
              role="tab"
              aria-selected={sort === s.key}
              className={sort === s.key ? "on" : ""}
              title={s.title}
              onClick={() => setSort(s.key)}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>
      <div
        className="thread-list"
        ref={listRef}
        onMouseEnter={() => setFrozen(shown.map((t) => t.thread_id))}
        onMouseLeave={() => setFrozen(null)}
        onKeyDown={onListKey}
      >
        {threads.length === 0 && <p className="muted small pad">No posts yet.</p>}
        {threads.length > 0 && shown.length === 0 && <p className="muted small pad">No threads match “{query}”.</p>}
        {shown.map((t) => (
          <div key={t.thread_id} className="thread-item">
          <div className="voter">
            <button
              className={`vote ${myVotes[t.thread_id] === 1 ? "on" : ""}`}
              title="Upvote: move this thread up the board (agents see the board in this order)"
              aria-label="Upvote"
              onClick={() => vote(t.thread_id, 1)}
            >
              ▲
            </button>
            <span className="small">{t.votes ?? 0}</span>
            <button
              className={`vote ${myVotes[t.thread_id] === -1 ? "on" : ""}`}
              title="Downvote: move this thread down the board"
              aria-label="Downvote"
              onClick={() => vote(t.thread_id, -1)}
            >
              ▼
            </button>
          </div>
          <button
            className={`thread-row ${isUnread(t) ? "unread" : ""} ${t.author_id === GOD_ID ? "god" : ""}`}
            onClick={() => openThread(t.thread_id)}
          >
            <span className="thread-row-top">
              <span className="thread-row-title">{t.title}</span>
              <span className="thread-row-time" title={new Date(t.last_activity * 1000).toLocaleString()}>
                {ago(t.last_activity, now)}
              </span>
            </span>
            <span className="thread-row-meta">
              <Dot color={agentColor(t.author_id)} />
              <span className="thread-row-author">{t.author}</span>
              <span className="sep">·</span>
              <span>
                {t.replies} {t.replies === 1 ? "reply" : "replies"}
              </span>
              {t.participants > 1 && (
                <>
                  <span className="sep">·</span>
                  <span>{t.participants} agents</span>
                </>
              )}
            </span>
            {t.latest && (
              <span className="thread-row-latest">
                <span className="author">{t.latest.author}:</span> {t.latest.body}
              </span>
            )}
          </button>
          </div>
        ))}
        {totalThreads > threads.length && (
          <p className="muted small pad center">Showing the {threads.length} most recently active threads.</p>
        )}
      </div>
    </aside>
  );
}

interface ThreadViewProps {
  runId: string;
  threadId: number;
  digest: ThreadDigest | undefined;
  version: number;
  now: number;
  agentName: (id: number) => string;
  agentColor: (id: number) => string;
  onSelectAgent: (id: number) => void;
  onSeen: (id: number, ts: number) => void;
  onBack: () => void;
  active: boolean;
}

interface Node {
  post: Post;
  children: Node[];
}

function ThreadView({ runId, threadId, digest, version, now, agentName, agentColor, onSelectAgent, onSeen, onBack, active }: ThreadViewProps) {
  const [posts, setPosts] = useState<Post[] | null>(null);
  const [missing, setMissing] = useState(false);
  const [collapsed, setCollapsed] = useState<Set<number>>(() => new Set());
  const [fresh, setFresh] = useState<Set<number>>(() => new Set());
  const [unseen, setUnseen] = useState<number[]>([]);
  const jumpTarget = useRef<number | null>(null);
  const activeRef = useRef(active);
  const scrollRef = useRef<HTMLDivElement>(null);
  const known = useRef<Set<number> | null>(null);
  const atBottom = useRef(false);

  useEffect(() => {
    let live = true;
    api
      .thread(runId, threadId)
      .then((p) => {
        if (!live) return;
        const el = scrollRef.current;
        if (activeRef.current) atBottom.current = !!el && el.scrollHeight - el.scrollTop - el.clientHeight < 24;
        const prev = known.current;
        known.current = new Set(p.map((x) => x.id));
        if (prev) {
          const added = p.filter((x) => !prev.has(x.id)).map((x) => x.id);
          if (added.length) {
            setFresh(new Set(added));
            setUnseen((u) => [...u, ...added.filter((id) => !u.includes(id))]);
          }
        }
        setPosts(p);
        setMissing(false);
      })
      .catch(() => live && setMissing(true));
    return () => {
      live = false;
    };
  }, [runId, threadId, version]);

  useEffect(() => {
    activeRef.current = active;
  }, [active]);

  useEffect(() => {
    if (!active || !posts?.length) return;
    onSeen(threadId, Math.max(...posts.map((p) => p.created_at)));
  }, [posts, threadId, onSeen, active]);

  // New replies stay "unseen" until their card has actually been on screen.
  const pruneSeen = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    setUnseen((u) => {
      const left = u.filter((id) => {
        const node = postNode(el, id);
        return !node || !isVisible(el, node);
      });
      return left.length === u.length ? u : left;
    });
  }, []);

  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el || !active) return;
    if (jumpTarget.current !== null) {
      postNode(el, jumpTarget.current)?.scrollIntoView({ behavior: "smooth", block: "center" });
      jumpTarget.current = null;
    } else if (atBottom.current && fresh.size) {
      el.scrollTop = el.scrollHeight;
    }
    const raf = requestAnimationFrame(pruneSeen);
    return () => cancelAnimationFrame(raf);
  }, [posts, fresh, collapsed, active, pruneSeen]);

  useEffect(() => {
    if (fresh.size === 0) return;
    const t = window.setTimeout(() => setFresh(new Set()), 2500);
    return () => window.clearTimeout(t);
  }, [fresh]);

  useEffect(() => {
    if (!active) return;
    const onKey = (e: globalThis.KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (e.key === "Escape" && tag !== "INPUT" && tag !== "TEXTAREA") onBack();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onBack, active]);

  const tree = useMemo(() => {
    if (!posts?.length) return null;
    const nodes = new Map<number, Node>(posts.map((p) => [p.id, { post: p, children: [] }]));
    const root = nodes.get(posts[0].id)!;
    for (const p of posts.slice(1)) {
      const parent = (p.parent_id !== null && nodes.get(p.parent_id)) || root;
      parent.children.push(nodes.get(p.id)!);
    }
    return root;
  }, [posts]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    atBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
    pruneSeen();
  };

  const jumpToNext = () => {
    const el = scrollRef.current;
    if (!el || !tree) return;
    const want = new Set(unseen);
    const find = (n: Node, path: number[]): number[] | null => {
      if (want.has(n.post.id)) return [...path, n.post.id];
      for (const c of n.children) {
        const hit = find(c, [...path, n.post.id]);
        if (hit) return hit;
      }
      return null;
    };
    const hit = find(tree, []);
    if (!hit) return setUnseen([]);
    const id = hit[hit.length - 1];
    const closed = hit.slice(0, -1).filter((a) => collapsed.has(a));
    if (closed.length) {
      jumpTarget.current = id;
      setCollapsed((s) => new Set([...s].filter((a) => !closed.includes(a))));
    } else {
      postNode(el, id)?.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  };

  const toggle = (id: number) =>
    setCollapsed((s) => {
      const n = new Set(s);
      if (!n.delete(id)) n.add(id);
      return n;
    });

  const root = tree?.post;
  const title = root?.title ?? digest?.title ?? "Thread";
  const participants = posts ? new Set(posts.map((p) => p.author_id)).size : (digest?.participants ?? 0);
  const replies = posts ? posts.length - 1 : (digest?.replies ?? 0);

  return (
    <>
      <header className="thread-head">
        <button className="ghost back" onClick={onBack} title="Back to all threads (Esc)" aria-label="Back to all threads">
          ←
        </button>
        <div className="thread-head-main">
          <h2 className="thread-head-title">{title}</h2>
          <span className="muted small">
            {replies} {replies === 1 ? "reply" : "replies"} · {participants} {participants === 1 ? "agent" : "agents"}
            {root && <> · started {ago(root.created_at, now)}</>}
          </span>
        </div>
      </header>
      <div className="thread-scroll" ref={scrollRef} onScroll={onScroll}>
        {missing && <p className="muted small pad">This thread is gone (the board may have been cleared).</p>}
        {!missing && !tree && <p className="muted small pad">Loading…</p>}
        {tree && (
          <>
            <article className={`op ${tree.post.author_id === GOD_ID ? "god" : ""}`}>
              <PostHead post={tree.post} now={now} agentName={agentName} agentColor={agentColor} onSelectAgent={onSelectAgent} />
              <p className="post-body">{withMentions(tree.post.body)}</p>
            </article>
            {tree.children.length === 0 ? (
              <p className="muted small pad">No replies yet.</p>
            ) : (
              <div className="reply-tree">
                {tree.children.map((n) => (
                  <Reply
                    key={n.post.id}
                    node={n}
                    depth={0}
                    parentAuthor={null}
                    now={now}
                    collapsed={collapsed}
                    fresh={fresh}
                    onToggle={toggle}
                    agentName={agentName}
                    agentColor={agentColor}
                    onSelectAgent={onSelectAgent}
                  />
                ))}
              </div>
            )}
          </>
        )}
      </div>
      {unseen.length > 0 && (
        <button className="new-below" onClick={jumpToNext} title="Jump to the next new reply you haven't seen">
          {unseen.length} new {unseen.length === 1 ? "reply" : "replies"}
        </button>
      )}
    </>
  );
}

const MAX_INDENT = 3;

interface ReplyProps {
  node: Node;
  depth: number;
  parentAuthor: string | null;
  now: number;
  collapsed: Set<number>;
  fresh: Set<number>;
  onToggle: (id: number) => void;
  agentName: (id: number) => string;
  agentColor: (id: number) => string;
  onSelectAgent: (id: number) => void;
}

function Reply({ node, depth, parentAuthor, now, collapsed, fresh, onToggle, agentName, agentColor, onSelectAgent }: ReplyProps) {
  const { post, children } = node;
  const isCollapsed = collapsed.has(post.id);
  const flattened = depth >= MAX_INDENT;
  return (
    <div className={`reply ${flattened ? "flat" : ""}`}>
      <div data-post={post.id} className={`reply-card ${fresh.has(post.id) ? "fresh" : ""} ${post.author_id === GOD_ID ? "god" : ""}`}>
        <PostHead post={post} now={now} agentName={agentName} agentColor={agentColor} onSelectAgent={onSelectAgent}>
          {flattened && parentAuthor && <span className="reply-to">↳ {parentAuthor}</span>}
        </PostHead>
        <p className="post-body">{withMentions(post.body)}</p>
        {children.length > 0 && (
          <button className={`ghost toggle ${isCollapsed ? "on" : ""}`} onClick={() => onToggle(post.id)} aria-expanded={!isCollapsed}>
            {isCollapsed ? `▸ Show ${countDescendants(node)} more` : "▾ Hide replies"}
          </button>
        )}
      </div>
      {!isCollapsed && children.length > 0 && (
        <div className={flattened ? "reply-children flat" : "reply-children"}>
          {children.map((c) => (
            <Reply
              key={c.post.id}
              node={c}
              depth={depth + 1}
              parentAuthor={agentName(post.author_id)}
              now={now}
              collapsed={collapsed}
              fresh={fresh}
              onToggle={onToggle}
              agentName={agentName}
              agentColor={agentColor}
              onSelectAgent={onSelectAgent}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function PostHead({
  post,
  now,
  agentName,
  agentColor,
  onSelectAgent,
  children,
}: {
  post: Post;
  now: number;
  agentName: (id: number) => string;
  agentColor: (id: number) => string;
  onSelectAgent: (id: number) => void;
  children?: ReactNode;
}) {
  const isGod = post.author_id === GOD_ID;
  return (
    <div className="post-head">
      <Dot color={agentColor(post.author_id)} />
      {isGod ? (
        <span className="author">{agentName(post.author_id)}</span>
      ) : (
        <button className="author link" onClick={() => onSelectAgent(post.author_id)} title="Open this agent">
          {agentName(post.author_id)}
        </button>
      )}
      {children}
      <span className="post-time" title={new Date(post.created_at * 1000).toLocaleString()}>
        {ago(post.created_at, now)}
      </span>
    </div>
  );
}

function postNode(container: HTMLElement, id: number): HTMLElement | null {
  return container.querySelector<HTMLElement>(`[data-post="${id}"]`);
}

function isVisible(container: HTMLElement, node: HTMLElement): boolean {
  const c = container.getBoundingClientRect();
  const r = node.getBoundingClientRect();
  return r.bottom > c.top && r.top < c.bottom;
}

function Dot({ color }: { color: string }) {
  return <span className="dot" style={{ background: color }} />;
}

function countDescendants(n: Node): number {
  return n.children.reduce((sum, c) => sum + 1 + countDescendants(c), 0);
}

function withMentions(text: string): ReactNode[] {
  return text.split(/(@[\w.-]+)/g).map((part, i) =>
    i % 2 === 1 ? (
      <span key={i} className="mention">
        {part}
      </span>
    ) : (
      part
    ),
  );
}

function ago(ts: number, now: number): string {
  const s = Math.max(0, Math.round(now - ts));
  if (s < 10) return "now";
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h`;
  return `${Math.floor(s / 86400)}d`;
}

function useNow(intervalMs = 10000): number {
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now() / 1000), intervalMs);
    return () => window.clearInterval(t);
  }, [intervalMs]);
  return now;
}

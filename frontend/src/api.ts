export interface RunConfig {
  width: number;
  height: number;
  agents: number;
  cooldown: number;
  jitter: number;
  pixels_per_turn: number;
  posts_per_turn: number;
  max_concurrency: number;
  seed: number | null;
  policy: "llm" | "llm-guided" | "random";
  narrator_interval: number;
  narrator_canvas_image: boolean;
}

export const DEFAULT_CONFIG: RunConfig = {
  width: 512,
  height: 320,
  agents: 100,
  cooldown: 10,
  jitter: 0.3,
  pixels_per_turn: 40,
  posts_per_turn: 1,
  max_concurrency: 10,
  seed: null,
  policy: "llm",
  narrator_interval: 60,
  narrator_canvas_image: true,
};

export type RunState = "created" | "running" | "paused" | "stopped";

export interface RunStats {
  pixels: number;
  posts: number;
  turns: number;
  errors: number;
}

export interface LatencySummary {
  n: number;
  mean: number;
  p50: number;
  p95: number;
  max: number;
}
export interface LLMStats {
  calls: number;
  tool_calls: number;
  tool_errors: number;
  no_tools: number;
  failures: number;
  latency?: { model: LatencySummary | null; turn: LatencySummary | null };
}

export interface NarratorStats {
  updates: number;
  answers: number;
  failures: number;
}

export interface RunSummary {
  id: string;
  state: RunState;
  config: RunConfig;
  created_at: number;
  started_at: number | null;
  stats: RunStats;
  agents: number;
  threads: number;
  clients: number;
  llm: LLMStats | null;
  narrator: NarratorStats;
}

export type AgentStatus = "idle" | "thinking" | "acting";

export interface Thought {
  ts: number;
  thought: string;
}

export interface Agent {
  id: number;
  name: string;
  country: string;
  color: number;
  home_x: number;
  home_y: number;
  cooldown: number;
  status: AgentStatus;
  last_thought: Thought | null;
  persona: string;
  persona_card: Record<string, string | number | string[] | null>;
}

export interface AgentDetail extends Agent {
  thoughts: Thought[];
  system_prompt: string | null;
}

export interface ThreadDigest {
  thread_id: number;
  title: string;
  author: string;
  author_id: number;
  created_at: number;
  last_activity: number;
  participants: number;
  replies: number;
  votes: number;
  latest: { author: string; author_id: number; body: string } | null;
}

export interface Post {
  id: number;
  author_id: number;
  thread_id: number;
  parent_id: number | null;
  title: string | null;
  body: string;
  created_at: number;
}

export type NarratorHighlight =
  | { type: "agent"; id: number; label: string }
  | { type: "thread"; id: number; label: string }
  | {
      type: "region";
      x: number;
      y: number;
      w: number;
      h: number;
      label: string;
    };

export interface NarratorUpdate {
  id: number;
  ts: number;
  kind: "update" | "answer";
  question: string | null;
  headline: string;
  bullets: string[];
  highlights: NarratorHighlight[];
  events?: { kind: string; text: string }[];
  source: "model" | "fallback";
  error: string | null;
  window: { seconds: number; turns: number; pixels: number; posts: number };
}

export interface NarratorState {
  interval: number;
  model: boolean;
  next_update_at: number | null;
  stats: NarratorStats;
  updates: NarratorUpdate[];
}

export interface Snapshot {
  type: "snapshot";
  run: RunSummary;
  palette: string[];
  agents: Agent[];
  threads: ThreadDigest[];
  narrator: NarratorState;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = sessionStorage.getItem("godToken");
  const headers: Record<string, string> = {
    "content-type": "application/json",
  };
  if (token) headers["X-God-Token"] = token;
  const res = await fetch(path, { headers, ...init });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = JSON.stringify((await res.json()).detail ?? detail);
    } catch {
      /* not json */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

export const GOD_ID = 65535;
export const GOD_NAME = "The Hand of God";

export const api = {
  listRuns: () => request<RunSummary[]>("/runs"),
  createRun: (config: RunConfig) =>
    request<RunSummary>("/runs", {
      method: "POST",
      body: JSON.stringify(config),
    }),
  getRun: (id: string) => request<RunSummary>(`/runs/${id}`),
  pause: (id: string) =>
    request<RunSummary>(`/runs/${id}/pause`, { method: "POST" }),
  resume: (id: string) =>
    request<RunSummary>(`/runs/${id}/resume`, { method: "POST" }),
  stop: (id: string) =>
    request<RunSummary>(`/runs/${id}/stop`, { method: "POST" }),
  agent: (id: string, agentId: number) =>
    request<AgentDetail>(`/runs/${id}/agents/${agentId}`),
  threads: (id: string) => request<ThreadDigest[]>(`/runs/${id}/threads`),
  thread: (id: string, threadId: number) =>
    request<Post[]>(`/runs/${id}/threads/${threadId}`),
  vote: (id: string, threadId: number, voter: string, vote: number) =>
    request<{ thread_id: number; votes: number }>(
      `/runs/${id}/threads/${threadId}/vote`,
      { method: "POST", body: JSON.stringify({ voter, vote }) },
    ),
  narrator: (id: string) => request<NarratorState>(`/runs/${id}/narrator`),
  ask: (id: string, question: string) =>
    request<NarratorUpdate>(`/runs/${id}/narrator/ask`, {
      method: "POST",
      body: JSON.stringify({ question }),
    }),
  godPixels: (id: string, pixels: [number, number, number][], erase = false) =>
    request<{ placed: number }>(`/runs/${id}/god/pixels`, {
      method: "POST",
      body: JSON.stringify({ pixels, erase }),
    }),
  godClearCanvas: (id: string) =>
    request<RunSummary>(`/runs/${id}/god/clear-canvas`, { method: "POST" }),
  godClearBoard: (id: string) =>
    request<RunSummary>(`/runs/${id}/god/clear-board`, { method: "POST" }),
  godAnnounce: (id: string, title: string, body: string) =>
    request<Post>(`/runs/${id}/god/announce`, {
      method: "POST",
      body: JSON.stringify({ title, body }),
    }),
  godPace: (
    id: string,
    pace: { cooldown?: number; pixels_per_turn?: number },
  ) =>
    request<RunSummary>(`/runs/${id}/god/pace`, {
      method: "POST",
      body: JSON.stringify(pace),
    }),
};

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
  policy: "llm" | "random";
}

export const DEFAULT_CONFIG: RunConfig = {
  width: 512,
  height: 320,
  agents: 100,
  cooldown: 10,
  jitter: 0.3,
  pixels_per_turn: 32,
  posts_per_turn: 1,
  max_concurrency: 10,
  seed: null,
  policy: "llm",
};

export type RunState = "created" | "running" | "paused" | "stopped";

export interface RunStats {
  pixels: number;
  posts: number;
  turns: number;
  errors: number;
}

export interface LLMStats {
  calls: number;
  tool_calls: number;
  tool_errors: number;
  no_tools: number;
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
}

export interface AgentDetail extends Agent {
  thoughts: Thought[];
  system_prompt: string | null;
}

export interface ThreadDigest {
  thread_id: number;
  title: string;
  author: string;
  replies: number;
  latest: { author: string; body: string } | null;
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

export interface Snapshot {
  type: "snapshot";
  run: RunSummary;
  palette: string[];
  agents: Agent[];
  threads: ThreadDigest[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { headers: { "content-type": "application/json" }, ...init });
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

export const api = {
  listRuns: () => request<RunSummary[]>("/runs"),
  createRun: (config: RunConfig) =>
    request<RunSummary>("/runs", { method: "POST", body: JSON.stringify(config) }),
  getRun: (id: string) => request<RunSummary>(`/runs/${id}`),
  pause: (id: string) => request<RunSummary>(`/runs/${id}/pause`, { method: "POST" }),
  resume: (id: string) => request<RunSummary>(`/runs/${id}/resume`, { method: "POST" }),
  stop: (id: string) => request<RunSummary>(`/runs/${id}/stop`, { method: "POST" }),
  agent: (id: string, agentId: number) => request<AgentDetail>(`/runs/${id}/agents/${agentId}`),
  threads: (id: string) => request<ThreadDigest[]>(`/runs/${id}/threads`),
  thread: (id: string, threadId: number) => request<Post[]>(`/runs/${id}/threads/${threadId}`),
};

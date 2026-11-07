export interface RunConfig {
  width: number;
  height: number;
  agents: number;
  cooldown: number;
  jitter: number;
  pixels_per_turn: number;
  pixel_interval: number;
  posts_per_turn: number;
  max_concurrency: number;
  seed: number | null;
  policy: "llm" | "random";
  narrator_interval: number;
  narrator_canvas_image: boolean;
  stage_auto: boolean;
  stage_lines: number;
  stage_vote_seconds: number;
}

export const DEFAULT_CONFIG: RunConfig = {
  width: 512,
  height: 320,
  agents: 100,
  cooldown: 3,
  jitter: 0.3,
  pixels_per_turn: 40,
  pixel_interval: 0.1,
  posts_per_turn: 1,
  max_concurrency: 10,
  seed: null,
  policy: "llm",
  narrator_interval: 60,
  narrator_canvas_image: true,
  stage_auto: false,
  stage_lines: 6,
  stage_vote_seconds: 20,
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
  stage?: { phase: StagePhase; debate_id: number };
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
}

export interface AgentDetail extends Agent {
  persona: string;
  persona_card: Record<string, string | number | string[] | null>;
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

export interface WorldNode {
  country: string;
  agents: number;
  /** Pixels painted by this country over the whole run (not what it holds now). */
  pixels: number;
  posts: number;
}

export interface WorldEdge {
  a: string;
  b: string;
  /** Replies and @mentions between the two countries. */
  talk: number;
  /** Pixels of b that a painted over, and vice versa. */
  raids_ab: number;
  raids_ba: number;
}

export interface WorldGraph {
  nodes: WorldNode[];
  edges: WorldEdge[];
}

export interface Snapshot {
  type: "snapshot";
  run: RunSummary;
  palette: string[];
  agents: Agent[];
  threads: ThreadDigest[];
  narrator: NarratorState;
}

export type StagePhase = "idle" | "debating" | "voting";
export type StageReaction = "cheer" | "boo" | "laugh" | "gasp";

export interface Debater {
  id: number;
  name: string;
  country: string;
  color: number;
  color_hex: string;
  podium: number;
  voice: string;
}

export interface StageLine {
  id: number;
  debate_id: number;
  kind: "debate" | "narrator" | "verdict";
  speaker_id: number;
  name: string;
  country: string;
  color: number | null;
  color_hex: string | null;
  podium: number | null;
  text: string;
  emotion: string;
  voice: string;
  audio: boolean;
  sample_rate: number;
  duration: number;
  ts: number;
}

export interface StageState {
  phase: StagePhase;
  auto: boolean;
  debate_id: number;
  topic: string;
  region: { x: number; y: number; w: number; h: number } | null;
  debaters: Debater[];
  votes: Record<string, number>;
  vote_ends_at: number | null;
  lines_per_debate: number;
  spoken: number;
  speech: boolean;
  model: boolean;
  stats: Record<string, number>;
  transcript?: StageLine[];
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
  world: (id: string) => request<WorldGraph>(`/runs/${id}/world`),
  thread: (id: string, threadId: number) =>
    request<Post[]>(`/runs/${id}/threads/${threadId}`),
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
    pace: { cooldown?: number; pixels_per_turn?: number; pixel_interval?: number },
  ) =>
    request<RunSummary>(`/runs/${id}/god/pace`, {
      method: "POST",
      body: JSON.stringify(pace),
    }),
  stage: (id: string) => request<StageState>(`/runs/${id}/stage`),
  stageStart: (id: string, opts: { a?: number; b?: number; topic?: string; auto?: boolean } = {}) =>
    request<StageState>(`/runs/${id}/stage/start`, { method: "POST", body: JSON.stringify(opts) }),
  stageStop: (id: string) => request<StageState>(`/runs/${id}/stage/stop`, { method: "POST" }),
  stageAuto: (id: string, auto: boolean) =>
    request<StageState>(`/runs/${id}/stage/auto`, { method: "POST", body: JSON.stringify({ auto }) }),
  stageVote: (id: string, voter: string, agentId: number) =>
    request<{ votes: Record<string, number> }>(`/runs/${id}/stage/vote`, {
      method: "POST",
      body: JSON.stringify({ voter, agent_id: agentId }),
    }),
  stageReact: (id: string, reaction: StageReaction, agentId: number | null) =>
    request<void>(`/runs/${id}/stage/react`, {
      method: "POST",
      body: JSON.stringify({ reaction, agent_id: agentId }),
    }),
  stageNarrate: (id: string, text: string, name: string, announce = false) =>
    request<{ queued: boolean }>(`/runs/${id}/stage/narrate`, {
      method: "POST",
      body: JSON.stringify({ text, name: name || null, announce }),
    }),
};

export const stageAudioUrl = (id: string, lineId: number) => `/runs/${id}/stage/lines/${lineId}/audio`;

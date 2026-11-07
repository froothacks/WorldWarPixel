import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type Agent,
  type NarratorHighlight,
  type NarratorState,
  type NarratorUpdate,
  type RunSummary,
  type ThreadDigest,
  type Thought,
} from "./api";
import { openRunStream, type ServerEvent } from "./ws";
import { PixelStore } from "./pixelStore";
import CanvasView, { type Region } from "./CanvasView";
import AgentList from "./AgentList";
import AgentDrawer from "./AgentDrawer";
import Board from "./Board";
import NarratorDock from "./NarratorDock";

interface Props {
  run: RunSummary;
  onLeave: () => void;
}

const MAX_LIVE_THOUGHTS = 20;
const MAX_NARRATOR_UPDATES = 50;

export default function LiveView({ run: initial, onLeave }: Props) {
  const [run, setRun] = useState<RunSummary>(initial);
  const [palette, setPalette] = useState<string[]>([]);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [threads, setThreads] = useState<ThreadDigest[]>([]);
  const [boardVersion, setBoardVersion] = useState(0);
  const [thoughts, setThoughts] = useState<Map<number, Thought[]>>(new Map());
  const [selected, setSelected] = useState<number | null>(null);
  const [openThread, setOpenThread] = useState<number | null>(null);
  const [focus, setFocus] = useState<Region | null>(null);
  const [narrator, setNarrator] = useState<NarratorState | null>(null);
  const [store, setStore] = useState<PixelStore | null>(null);
  const [version, setVersion] = useState(0);
  const [connection, setConnection] = useState("connecting…");
  const paletteRef = useRef<string[]>([]);
  const storeRef = useRef<PixelStore | null>(null);
  const refreshTimer = useRef<number | null>(null);

  const scheduleThreadRefresh = useCallback(() => {
    if (refreshTimer.current !== null) return;
    refreshTimer.current = window.setTimeout(() => {
      refreshTimer.current = null;
      api.threads(initial.id).then(setThreads).catch(() => undefined);
      setBoardVersion((v) => v + 1);
    }, 500);
  }, [initial.id]);

  useEffect(() => {
    const onEvent = (ev: ServerEvent) => {
      switch (ev.type) {
        case "snapshot":
          paletteRef.current = ev.palette;
          setPalette(ev.palette);
          setRun(ev.run);
          setAgents(ev.agents);
          setThreads(ev.threads);
          setNarrator(ev.narrator);
          setConnection("live");
          break;
        case "narrator.update": {
          const { type: _type, ...update } = ev;
          setNarrator((n) => (n ? { ...n, updates: appendUpdate(n.updates, update) } : n));
          break;
        }
        case "run.state":
          setRun((r) => ({ ...r, state: ev.state, stats: ev.stats }));
          break;
        case "agent.status":
          setAgents((list) => list.map((a) => (a.id === ev.agent_id ? { ...a, status: ev.status } : a)));
          break;
        case "agent.thought":
          setThoughts((m) => {
            const next = new Map(m);
            const prev = next.get(ev.agent_id) ?? [];
            next.set(ev.agent_id, [{ ts: ev.ts, thought: ev.thought }, ...prev].slice(0, MAX_LIVE_THOUGHTS));
            return next;
          });
          break;
        case "post.created":
          scheduleThreadRefresh();
          break;
      }
    };
    const close = openRunStream(initial.id, {
      onEvent,
      onCanvas: (w, h, pixels, owners) => {
        const s = new PixelStore(w, h, paletteRef.current);
        s.loadSnapshot(pixels, owners);
        storeRef.current = s;
        setStore(s);
        setVersion(s.version);
      },
      onPixels: (writes) => {
        const s = storeRef.current;
        if (!s) return;
        s.apply(writes);
        setVersion(s.version);
      },
      onClose: (reason) => setConnection(`disconnected: ${reason}`),
    });
    return () => {
      close();
      if (refreshTimer.current !== null) window.clearTimeout(refreshTimer.current);
    };
  }, [initial.id, scheduleThreadRefresh]);

  // Stats come with run.state events; poll a little for the counters in between.
  useEffect(() => {
    if (run.state === "stopped") return;
    const t = window.setInterval(() => api.getRun(initial.id).then(setRun).catch(() => undefined), 2000);
    return () => window.clearInterval(t);
  }, [initial.id, run.state]);

  const agentMap = useMemo(() => new Map(agents.map((a) => [a.id, a])), [agents]);
  const agentName = useCallback((id: number) => agentMap.get(id)?.name ?? `agent ${id}`, [agentMap]);
  const lastThought = useMemo(() => {
    const m = new Map<number, string>();
    thoughts.forEach((list, id) => {
      if (list[0]) m.set(id, list[0].thought);
    });
    return m;
  }, [thoughts]);

  const selectedAgent = selected !== null ? agentMap.get(selected) : undefined;
  const pixelsOwned = useMemo(
    () => (selected !== null && store ? store.countOwned(selected) : 0),
    // version is what makes this recount as pixels land.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [selected, store, version],
  );

  const onHighlight = useCallback((h: NarratorHighlight) => {
    switch (h.type) {
      case "agent":
        setSelected(h.id);
        break;
      case "thread":
        setSelected(null);
        setOpenThread(h.id);
        break;
      case "region":
        setFocus((f) => ({ x: h.x, y: h.y, w: h.w, h: h.h, nonce: (f?.nonce ?? 0) + 1 }));
        break;
    }
  }, []);

  const ask = useCallback(
    async (question: string) => {
      const update = await api.ask(initial.id, question);
      // The socket delivers the same update; appendUpdate dedupes by id.
      setNarrator((n) => (n ? { ...n, updates: appendUpdate(n.updates, update) } : n));
    },
    [initial.id],
  );

  const control = (action: "pause" | "resume" | "stop") => api[action](initial.id).then(setRun).catch((e: Error) => alert(e.message));

  return (
    <div className="live">
      <header className="topbar">
        <button className="ghost" onClick={onLeave}>← Runs</button>
        <strong>WorldWarPixel</strong>
        <span className={`state state-${run.state}`}>{run.state}</span>
        <span className="muted small">
          run {run.id} · {run.agents} agents · {run.stats.turns.toLocaleString()} turns · {run.stats.pixels.toLocaleString()} px · {run.stats.posts.toLocaleString()} posts
          {run.llm && ` · ${run.llm.tool_calls.toLocaleString()} tool calls / ${run.llm.calls.toLocaleString()} model calls${run.llm.failures ? ` · ${run.llm.failures} failed` : ""}`}
          {" · "}{connection}
        </span>
        <span className="spacer" />
        {run.state === "running" && <button onClick={() => control("pause")}>Pause</button>}
        {run.state === "paused" && <button onClick={() => control("resume")}>Resume</button>}
        {run.state !== "stopped" && <button className="danger" onClick={() => control("stop")}>Stop</button>}
      </header>
      <div className="live-grid">
        <AgentList agents={agents} palette={palette} selected={selected} lastThought={lastThought} onSelect={setSelected} />
        <CanvasView store={store} version={version} agents={agentMap} highlightAgent={selected} onPickAgent={setSelected} focus={focus} />
        {selectedAgent ? (
          <AgentDrawer
            runId={initial.id}
            agent={selectedAgent}
            palette={palette}
            liveThoughts={thoughts.get(selectedAgent.id) ?? []}
            pixelsOwned={pixelsOwned}
            onClose={() => setSelected(null)}
          />
        ) : (
          <Board runId={initial.id} threads={threads} agentName={agentName} version={boardVersion} totalThreads={run.threads} open={openThread} onOpen={setOpenThread} />
        )}
      </div>
      <NarratorDock narrator={narrator} stats={run.stats} threads={run.threads} onHighlight={onHighlight} onAsk={ask} />
    </div>
  );
}

function appendUpdate(list: NarratorUpdate[], update: NarratorUpdate): NarratorUpdate[] {
  if (list.some((u) => u.id === update.id)) return list;
  return [...list, update].slice(-MAX_NARRATOR_UPDATES);
}

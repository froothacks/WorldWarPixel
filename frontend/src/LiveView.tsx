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
import GodBar, { type GodTool } from "./GodBar";
import { GOD_ID, GOD_NAME } from "./api";
import CanvasView, { type Region } from "./CanvasView";
import AgentList from "./AgentList";
import AgentDrawer from "./AgentDrawer";
import Board from "./Board";
import NarratorDock from "./NarratorDock";
import Timelapse from "./Timelapse";

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
      api
        .threads(initial.id)
        .then(setThreads)
        .catch(() => undefined);
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
          const { type: _type, stats, ...update } = ev;
          setNarrator((n) =>
            n
              ? {
                  ...n,
                  stats: stats ?? n.stats,
                  updates: appendUpdate(n.updates, update),
                }
              : n,
          );
          break;
        }
        case "run.state":
          setRun((r) => ({ ...r, state: ev.state, stats: ev.stats }));
          break;
        case "agent.status":
          setAgents((list) =>
            list.map((a) =>
              a.id === ev.agent_id ? { ...a, status: ev.status } : a,
            ),
          );
          break;
        case "agent.thought":
          setThoughts((m) => {
            const next = new Map(m);
            const prev = next.get(ev.agent_id) ?? [];
            next.set(
              ev.agent_id,
              [{ ts: ev.ts, thought: ev.thought }, ...prev].slice(
                0,
                MAX_LIVE_THOUGHTS,
              ),
            );
            return next;
          });
          break;
        case "post.created":
        case "thread.voted":
          scheduleThreadRefresh();
          break;
        case "board.cleared":
          setThreads([]);
          setOpenThread(null);
          setBoardVersion((v) => v + 1);
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
      if (refreshTimer.current !== null)
        window.clearTimeout(refreshTimer.current);
    };
  }, [initial.id, scheduleThreadRefresh]);

  // Stats come with run.state events; poll a little for the counters in between.
  useEffect(() => {
    if (run.state === "stopped") return;
    const t = window.setInterval(
      () =>
        api
          .getRun(initial.id)
          .then(setRun)
          .catch(() => undefined),
      2000,
    );
    return () => window.clearInterval(t);
  }, [initial.id, run.state]);

  const agentMap = useMemo(
    () => new Map(agents.map((a) => [a.id, a])),
    [agents],
  );
  const agentName = useCallback(
    (id: number) =>
      id === GOD_ID ? GOD_NAME : (agentMap.get(id)?.name ?? `agent ${id}`),
    [agentMap],
  );
  const agentColor = useCallback(
    (id: number) => {
      const a = agentMap.get(id);
      return a ? (palette[a.color] ?? "#888") : id === GOD_ID ? "#c9a7ff" : "#888";
    },
    [agentMap, palette],
  );
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
        setFocus((f) => ({
          x: h.x,
          y: h.y,
          w: h.w,
          h: h.h,
          nonce: (f?.nonce ?? 0) + 1,
        }));
        break;
    }
  }, []);

  const ask = useCallback(
    async (question: string) => {
      const update = await api.ask(initial.id, question);
      // The socket delivers the same update; appendUpdate dedupes by id.
      setNarrator((n) =>
        n ? { ...n, updates: appendUpdate(n.updates, update) } : n,
      );
    },
    [initial.id],
  );

  const control = (action: "pause" | "resume" | "stop") =>
    api[action](initial.id)
      .then(setRun)
      .catch((e: Error) => alert(e.message));
  const [god, setGod] = useState(false);
  const [timelapse, setTimelapse] = useState(false);
  const [tool, setTool] = useState<GodTool | null>(null);
  const paint = useCallback(
    (pixels: [number, number, number][], erase: boolean) => {
      api.godPixels(initial.id, pixels, erase).catch(() => undefined);
    },
    [initial.id],
  );

  const llm = run.llm;
  const turnLatency = llm?.latency?.turn;
  const llmTitle = llm
    ? `${llm.tool_calls.toLocaleString()} tool calls / ${llm.calls.toLocaleString()} model calls${llm.failures ? ` · ${llm.failures} failed` : ""}` +
      (turnLatency ? ` · turn p50 ${turnLatency.p50.toFixed(2)}s / p95 ${turnLatency.p95.toFixed(2)}s` : "")
    : undefined;
  const live = connection === "live";

  return (
    <div className="live">
      <header className="topbar">
        <button className="brand" onClick={onLeave} title="Back to all runs">
          WorldWarPixel
        </button>
        <span className="crumb">
          <span className="slash">/</span>
          <span className="mono">{run.id}</span>
          <span className={`state state-${run.state}`}>{run.state}</span>
        </span>
        <dl className="stats" title={llmTitle}>
          <Stat k="agents" v={run.agents} />
          <Stat k="turns" v={run.stats.turns} />
          <Stat k="px" v={run.stats.pixels} />
          <Stat k="posts" v={run.stats.posts} />
          {llm && <Stat k="calls" v={llm.calls} className="opt" />}
          {turnLatency && <Stat k="p50" v={`${turnLatency.p50.toFixed(2)}s`} className="opt" />}
          {llm && llm.failures > 0 && <Stat k="failed" v={llm.failures} className="bad" />}
        </dl>
        <span className="spacer" />
        <span className={`conn ${live ? "conn-on" : ""}`} title={connection}>
          {live ? "● live" : connection.startsWith("disconnected") ? "○ offline" : "○ connecting"}
        </span>
        <div className="actions">
          <button
            className={god ? "on" : ""}
            onClick={() => {
              setGod(!god);
              if (god) setTool(null);
            }}
            title="Intervene: paint, erase, clear, announce, change the pace"
          >
            God mode
          </button>
          <button onClick={() => setTimelapse(true)} title="Replay the canvas history so far">
            Timelapse
          </button>
          {run.state === "running" && <button onClick={() => control("pause")}>Pause</button>}
          {run.state === "paused" && <button onClick={() => control("resume")}>Resume</button>}
          {run.state !== "stopped" && (
            <button className="danger" onClick={() => control("stop")}>
              Stop
            </button>
          )}
        </div>
      </header>
      <div className="live-grid">
        <section className="stage">
          {god ? (
            <GodBar
              runId={initial.id}
              run={run}
              palette={palette}
              tool={tool}
              onTool={setTool}
              onRun={setRun}
              onClose={() => {
                setGod(false);
                setTool(null);
              }}
            />
          ) : (
            <span />
          )}
          <CanvasView
            store={store}
            version={version}
            agents={agentMap}
            highlightAgent={selected}
            onPickAgent={setSelected}
            focus={focus}
            tool={god ? tool : null}
            onPaint={paint}
          />
          <NarratorDock narrator={narrator} paused={run.state !== "running"} onHighlight={onHighlight} onAsk={ask} />
        </section>
        <Board
          runId={initial.id}
          threads={threads}
          agentName={agentName}
          agentColor={agentColor}
          onSelectAgent={setSelected}
          version={boardVersion}
          totalThreads={run.threads}
          open={openThread}
          onOpen={setOpenThread}
          onVoted={scheduleThreadRefresh}
        />
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
          <AgentList agents={agents} palette={palette} selected={selected} lastThought={lastThought} onSelect={setSelected} />
        )}
      </div>
      {timelapse && <Timelapse runId={initial.id} palette={palette} onClose={() => setTimelapse(false)} />}
    </div>
  );
}

function Stat({ k, v, className = "" }: { k: string; v: number | string; className?: string }) {
  return (
    <div className={`stat ${className}`}>
      <dt>{k}</dt>
      <dd>{typeof v === "number" ? v.toLocaleString() : v}</dd>
    </div>
  );
}

function appendUpdate(
  list: NarratorUpdate[],
  update: NarratorUpdate,
): NarratorUpdate[] {
  if (list.some((u) => u.id === update.id)) return list;
  return [...list, update].slice(-MAX_NARRATOR_UPDATES);
}

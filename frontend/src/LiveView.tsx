import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type Agent,
  type NarratorHighlight,
  type NarratorState,
  type NarratorUpdate,
  type RunSummary,
  type StageState,
  type ThreadDigest,
  type Thought,
} from "./api";
import { openRunStream, type ServerEvent } from "./ws";
import { PixelStore } from "./pixelStore";
import GodBar, { type GodTool } from "./GodBar";
import DebateBar from "./DebateBar";
import { useDebate, useStageAudio } from "./debate";
import { DISPATCH_ID, DISPATCH_NAME, GOD_ID, GOD_NAME } from "./api";
import CanvasView, { type Region } from "./CanvasView";
import AgentList from "./AgentList";
import AgentDrawer from "./AgentDrawer";
import Territory from "./Territory";
import Alliances from "./Alliances";
import Board from "./Board";
import NarratorDock from "./NarratorDock";
import Timelapse from "./Timelapse";

interface Props {
  run: RunSummary;
  onLeave: () => void;
}

const MAX_LIVE_THOUGHTS = 20;
const MAX_NARRATOR_UPDATES = 50;

type SideTab = "agents" | "territory" | "alliances";
const TABS: [SideTab, string][] = [
  ["agents", "Agents"],
  ["territory", "Territory"],
  ["alliances", "Alliances"],
];

export default function LiveView({ run: initial, onLeave }: Props) {
  const [run, setRun] = useState<RunSummary>(initial);
  const [palette, setPalette] = useState<string[]>([]);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [threads, setThreads] = useState<ThreadDigest[]>([]);
  const [boardVersion, setBoardVersion] = useState(0);
  const [thoughts, setThoughts] = useState<Map<number, Thought[]>>(new Map());
  const [selected, setSelected] = useState<number | null>(null);
  const [tab, setTab] = useState<SideTab>("agents");
  const [country, setCountry] = useState<string | null>(null);
  const [openThread, setOpenThread] = useState<number | null>(null);
  const [focus, setFocus] = useState<Region | null>(null);
  const [debateOpen, setDebateOpen] = useState(false);
  const [stageSound, setStageSound] = useState(false);
  const [narrator, setNarrator] = useState<NarratorState | null>(null);
  const [store, setStore] = useState<PixelStore | null>(null);
  const [version, setVersion] = useState(0);
  const [connection, setConnection] = useState("connecting…");
  const paletteRef = useRef<string[]>([]);
  const storeRef = useRef<PixelStore | null>(null);
  const refreshTimer = useRef<number | null>(null);
  const focusedDebate = useRef<number | null>(null);
  const debate = useDebate(initial.id, false);
  const debateState = debate.state;
  const acceptDebate = debate.accept;
  useStageAudio(initial.id, debate.transcript.at(-1) ?? null, stageSound);

  useEffect(() => {
    const stage = debateState;
    if (!stage?.region || focusedDebate.current === stage.debate_id) return;
    focusedDebate.current = stage.debate_id;
    setFocus((prev) => ({
      ...stage.region!,
      nonce: (prev?.nonce ?? 0) + 1,
    }));
  }, [debateState]);

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
      acceptDebate(ev);
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
  }, [acceptDebate, initial.id, scheduleThreadRefresh]);

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
      id === DISPATCH_ID
        ? DISPATCH_NAME
        : id === GOD_ID
          ? GOD_NAME
          : (agentMap.get(id)?.name ?? `agent ${id}`),
    [agentMap],
  );
  const agentColor = useCallback(
    (id: number) => {
      const a = agentMap.get(id);
      return a
        ? (palette[a.color] ?? "#888")
        : id === DISPATCH_ID
          ? "#e0b341"
          : id === GOD_ID
            ? "#c9a7ff"
            : "#888";
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
  const highlightMask = useMemo(() => {
    const mask = new Uint8Array(65536);
    if (selected !== null) return null;
    if (country) {
      for (const a of agents) if (a.country === country) mask[a.id] = 1;
      return mask;
    }
    for (const debater of debate.state?.debaters ?? []) mask[debater.id] = 1;
    return mask.some(Boolean) ? mask : null;
  }, [country, agents, debate.state?.debaters, selected]);
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
  const say = useCallback(
    async (text: string, name: string) => {
      await api.stageNarrate(initial.id, text, name);
    },
    [initial.id],
  );
  const acceptStageState = useCallback(
    (state: StageState) => acceptDebate({ type: "stage.state", ...state }),
    [acceptDebate],
  );

  const control = (action: "pause" | "resume" | "stop") =>
    api[action](initial.id)
      .then(setRun)
      .catch((e: Error) => alert(e.message));
  const [god, setGod] = useState(false);
  const [timelapse, setTimelapse] = useState(false);
  const [tool, setTool] = useState<GodTool | null>(null);
  const paintRejected = useRef(false);
  const paint = useCallback(
    (pixels: [number, number, number][], erase: boolean) => {
      api.godPixels(initial.id, pixels, erase).then(
        () => {
          paintRejected.current = false;
        },
        (e: Error) => {
          // One alert per streak of failures, not one per brush segment.
          if (paintRejected.current) return;
          paintRejected.current = true;
          alert(`God brush rejected: ${e.message}`);
        },
      );
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
  // Roster runs play real attendees as individuals; each agent's "country" is its own name.
  const people = (run.config.roster ?? "world") !== "world";

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
          {people && <span className="roster-tag">{run.agents} attendees</span>}
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
              const next = !god;
              setGod(next);
              if (next) setDebateOpen(false);
              if (god) setTool(null);
            }}
            title="Intervene: paint, erase, clear, announce, change the pace"
          >
            God mode
          </button>
          <button
            className={debateOpen ? "on" : ""}
            onClick={() => {
              const next = !debateOpen;
              setDebateOpen(next);
              if (next) {
                setGod(false);
                setTool(null);
              }
            }}
            title="Open live debate controls"
          >
            Debate{debate.state && debate.state.phase !== "idle" ? " ●" : ""}
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
      {llm?.last_error && (
        <div className="errbar" role="alert">
          Model calls failing ({llm.failures}): {llm.last_error}
        </div>
      )}
      <div className="live-grid">
        <section className="stage">
          {debateOpen ? (
            <DebateBar
              runId={initial.id}
              state={debate.state}
              agents={agents}
              palette={palette}
              sound={stageSound}
              onSound={setStageSound}
              onSelectAgent={setSelected}
              onState={acceptStageState}
            />
          ) : god ? (
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
            highlightMask={highlightMask}
            onPickAgent={setSelected}
            focus={focus}
            tool={god ? tool : null}
            onPaint={paint}
          />
          <NarratorDock
            narrator={narrator}
            paused={run.state !== "running"}
            stageLines={debate.transcript}
            onHighlight={onHighlight}
            onAsk={ask}
            onSay={say}
          />
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
        />
        <div className="side">
          <nav className="tabs">
            {TABS.map(([id, label]) => (
              <button
                key={id}
                className={tab === id && !selectedAgent ? "on" : ""}
                onClick={() => {
                  setTab(id);
                  setSelected(null);
                }}
              >
                {people && id === "agents" ? "People" : label}
              </button>
            ))}
          </nav>
          {selectedAgent && (
            <AgentDrawer
              runId={initial.id}
              agent={selectedAgent}
              palette={palette}
              liveThoughts={thoughts.get(selectedAgent.id) ?? []}
              pixelsOwned={pixelsOwned}
              onClose={() => setSelected(null)}
            />
          )}
          <AgentList
            agents={agents}
            palette={palette}
            selected={selected}
            lastThought={lastThought}
            onSelect={setSelected}
            hidden={selectedAgent !== undefined || tab !== "agents"}
            people={people}
          />
          <Territory
            store={store}
            version={version}
            agents={agents}
            palette={palette}
            highlight={country}
            onHighlight={setCountry}
            hidden={selectedAgent !== undefined || tab !== "territory"}
            units={people ? "people" : "countries"}
          />
          <Alliances
            runId={initial.id}
            running={run.state === "running"}
            agents={agents}
            palette={palette}
            highlight={country}
            onHighlight={setCountry}
            hidden={selectedAgent !== undefined || tab !== "alliances"}
          />
        </div>
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

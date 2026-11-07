import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import {
  api,
  stageAudioUrl,
  type Agent,
  type StageLine,
  type StageState,
} from "./api";
import { PixelStore } from "./pixelStore";
import { openRunStream, type ServerEvent } from "./ws";

const TRANSCRIPT_MAX = 200;

interface DebateModel {
  state: StageState | null;
  transcript: StageLine[];
  verdict: { debate_id: number; winner_id: number | null } | null;
}

const INITIAL_MODEL: DebateModel = { state: null, transcript: [], verdict: null };

type DebateAction =
  | { type: "initial"; state: StageState }
  | { type: "event"; event: ServerEvent };

function reduceDebate(model: DebateModel, action: DebateAction): DebateModel {
  if (action.type === "initial") {
    return {
      ...model,
      state: model.state ?? action.state,
      transcript: model.transcript.length
        ? model.transcript
        : action.state.transcript ?? [],
    };
  }
  const event = action.event;
  if (event.type === "snapshot") {
    const state = event.stage;
    return {
      ...model,
      state: state ? { ...state, transcript: model.transcript } : model.state,
      transcript: model.transcript.length
        ? model.transcript
        : state?.transcript ?? [],
    };
  }
  if (event.type === "stage.state") {
    const { type: _type, ...state } = event;
    return {
      ...model,
      state: { ...state, transcript: model.transcript },
    };
  }
  if (event.type === "stage.line") {
    const { type: _type, ...line } = event;
    if (model.transcript.some((existing) => existing.id === line.id)) return model;
    const transcript = [...model.transcript, line].slice(-TRANSCRIPT_MAX);
    return {
      ...model,
      transcript,
      state: model.state ? { ...model.state, transcript } : null,
    };
  }
  if (event.type === "stage.votes") {
    return {
      ...model,
      state: model.state ? { ...model.state, votes: event.votes } : null,
    };
  }
  if (event.type === "stage.verdict") {
    return {
      ...model,
      verdict: { debate_id: event.debate_id, winner_id: event.winner_id },
      state: model.state ? { ...model.state, votes: event.votes } : null,
    };
  }
  return model;
}

export function useDebate(runId: string, listen = true) {
  const [model, dispatch] = useReducer(reduceDebate, INITIAL_MODEL);
  const [palette, setPalette] = useState<string[]>([]);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [store, setStore] = useState<PixelStore | null>(null);
  const [version, setVersion] = useState(0);
  const [live, setLive] = useState(false);
  const [headline, setHeadline] = useState<string | null>(null);
  const paletteRef = useRef<string[]>([]);
  const storeRef = useRef<PixelStore | null>(null);

  const accept = useCallback((event: ServerEvent) => {
    dispatch({ type: "event", event });
    if (event.type === "snapshot") {
      setHeadline(event.narrator.updates.at(-1)?.headline ?? null);
    } else if (event.type === "narrator.update") {
      setHeadline(event.headline);
    }
  }, []);

  useEffect(() => {
    let mounted = true;
    api
      .stage(runId)
      .then((state) => {
        if (mounted) dispatch({ type: "initial", state });
      })
      .catch(() => undefined);
    if (!listen) return () => { mounted = false; };
    const close = openRunStream(runId, {
      onEvent: (event) => {
        accept(event);
        if (event.type === "snapshot") {
          paletteRef.current = event.palette;
          setPalette(event.palette);
          setAgents(event.agents);
          setLive(true);
        } else if (event.type === "narrator.update") {
          setLive(true);
        }
      },
      onCanvas: (width, height, pixels, owners) => {
        const next = new PixelStore(width, height, paletteRef.current);
        next.loadSnapshot(pixels, owners);
        storeRef.current = next;
        setStore(next);
        setVersion(next.version);
      },
      onPixels: (writes) => {
        const current = storeRef.current;
        if (!current) return;
        current.apply(writes);
        setVersion(current.version);
      },
      onClose: () => setLive(false),
    });
    return () => {
      mounted = false;
      close();
    };
  }, [accept, listen, runId]);

  const agentMap = useMemo(() => new Map(agents.map((agent) => [agent.id, agent])), [agents]);
  return {
    ...model,
    accept,
    agentMap,
    agents,
    headline,
    live,
    palette,
    store,
    version,
  };
}

export function useDebateFocus(state: StageState | null) {
  const [focus, setFocus] = useState<{
    x: number;
    y: number;
    w: number;
    h: number;
    nonce: number;
  } | null>(null);
  const focusedDebate = useRef<number | null>(null);
  useEffect(() => {
    const region = state?.region;
    if (!state || !region || focusedDebate.current === state.debate_id) return;
    focusedDebate.current = state.debate_id;
    setFocus((previous) => ({
      ...region,
      nonce: (previous?.nonce ?? 0) + 1,
    }));
  }, [state]);
  return focus;
}

function speakWithBrowser(line: StageLine, done: () => void): boolean {
  if (typeof speechSynthesis === "undefined" || typeof SpeechSynthesisUtterance === "undefined") {
    return false;
  }
  const utterance = new SpeechSynthesisUtterance(line.text);
  const voices = speechSynthesis.getVoices().filter((voice) => voice.lang.startsWith("en"));
  if (voices.length) utterance.voice = voices[Math.abs(line.speaker_id) % voices.length];
  utterance.pitch = line.kind === "narrator" ? 1 : 0.8 + (Math.abs(line.speaker_id) % 5) * 0.1;
  utterance.rate = 1.05;
  utterance.onend = done;
  utterance.onerror = done;
  speechSynthesis.speak(utterance);
  return true;
}

export function useStageAudio(
  runId: string,
  line: StageLine | null,
  enabled: boolean,
  filter?: (line: StageLine) => boolean,
) {
  const queue = useRef<StageLine[]>([]);
  const lastQueuedId = useRef<number | null>(null);
  const audio = useRef<HTMLAudioElement | null>(null);
  const playing = useRef(false);
  const enabledRef = useRef(enabled);

  const playNext = useCallback(function advance() {
    if (!enabledRef.current) {
      queue.current = [];
      return;
    }
    if (playing.current) return;
    const next = queue.current.shift();
    if (!next) return;
    playing.current = true;
    let finished = false;
    const done = () => {
      if (finished) return;
      finished = true;
      playing.current = false;
      audio.current = null;
      advance();
    };
    if (next.audio) {
      const player = new Audio(stageAudioUrl(runId, next.id));
      audio.current = player;
      player.onended = done;
      player.onerror = () => {
        if (!speakWithBrowser(next, done)) done();
      };
      void player.play().catch(() => {
        if (!speakWithBrowser(next, done)) done();
      });
    } else if (!speakWithBrowser(next, done)) {
      window.setTimeout(done, next.duration * 1000);
    }
  }, [runId]);

  useEffect(() => {
    enabledRef.current = enabled;
    if (!enabled) {
      queue.current = [];
      audio.current?.pause();
      audio.current = null;
      playing.current = false;
      if (typeof speechSynthesis !== "undefined") speechSynthesis.cancel();
    }
  }, [enabled]);

  useEffect(() => {
    if (!enabled || !line || filter?.(line) === false || lastQueuedId.current === line.id) return;
    lastQueuedId.current = line.id;
    queue.current = [...queue.current.slice(-8), line];
    playNext();
  }, [enabled, filter, line, playNext]);

  useEffect(
    () => () => {
      audio.current?.pause();
      if (typeof speechSynthesis !== "undefined") speechSynthesis.cancel();
    },
    [],
  );
}

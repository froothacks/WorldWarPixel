import { useEffect, useState } from "react";
import { api, type ModelInfo, type RunSummary } from "./api";

export interface GodTool {
  mode: "paint" | "erase";
  color: number;
  size: number;
}

interface Props {
  runId: string;
  run: RunSummary;
  palette: string[];
  tool: GodTool | null;
  onTool: (tool: GodTool | null) => void;
  onRun: (run: RunSummary) => void;
  onError: (title: string, detail?: string) => void;
  onClose: () => void;
}

const SIZES = [1, 2, 4, 8, 16];

const ACTION_ERRORS: Record<string, string> = {
  announce: "Announcement failed",
  pace: "Couldn't change the pace",
  model: "Couldn't switch the model",
  clear: "Clear failed",
};

export default function GodBar({
  runId,
  run,
  palette,
  tool,
  onTool,
  onRun,
  onError,
  onClose,
}: Props) {
  const [color, setColor] = useState(5);
  const [size, setSize] = useState(4);
  const [busy, setBusy] = useState<string | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelChoice, setModelChoice] = useState(run.config.model);

  useEffect(() => {
    if (run.config.policy === "random") return;
    api
      .listModels()
      .then(setModels)
      .catch(() => setModels([]));
  }, [run.config.policy]);

  const act = async (label: string, fn: () => Promise<unknown>) => {
    setBusy(label);
    try {
      await fn();
    } catch (e) {
      onError(ACTION_ERRORS[label] ?? "God mode action failed", (e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const pick = (mode: "paint" | "erase") => {
    if (tool?.mode === mode) onTool(null);
    else onTool({ mode, color, size });
  };
  const setColorAnd = (c: number) => {
    setColor(c);
    if (tool) onTool({ ...tool, mode: "paint", color: c });
  };
  const setSizeAnd = (s: number) => {
    setSize(s);
    if (tool) onTool({ ...tool, size: s });
  };

  const [announcing, setAnnouncing] = useState(false);
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");
  const announce = () => {
    if (!title.trim() || !body.trim()) return;
    void act("announce", () =>
      api.godAnnounce(runId, title.trim(), body.trim()).then(() => {
        setAnnouncing(false);
        setTitle("");
        setBody("");
      }),
    );
  };
  const pace = (cooldown: number) =>
    void act("pace", () => api.godPace(runId, { cooldown }).then(onRun));
  const pxPerTurn = (pixels_per_turn: number) =>
    void act("pace", () => api.godPace(runId, { pixels_per_turn }).then(onRun));
  const pxInterval = (ms: number) =>
    void act("pace", () => api.godPace(runId, { pixel_interval: ms / 1000 }).then(onRun));

  return (
    <div className="godbar">
      <span className="label" title="You are the human in the loop. Everything here is visible to the agents.">
        God mode
      </span>
      <span className="sep" />
      <button
        className={tool?.mode === "paint" ? "on" : ""}
        onClick={() => pick("paint")}
        title="Paint on the canvas as The Hand of God"
      >
        Brush
      </button>
      <button
        className={tool?.mode === "erase" ? "on" : ""}
        onClick={() => pick("erase")}
        title="Erase pixels back to blank, unowned canvas"
      >
        Eraser
      </button>
      <span className="swatches">
        {palette.map((hex, i) => (
          <button
            key={i}
            className={`swatch ${color === i && tool?.mode !== "erase" ? "on" : ""}`}
            style={{ background: hex }}
            title={`colour ${i}`}
            onClick={() => setColorAnd(i)}
          />
        ))}
      </span>
      <label className="small">
        size
        <select value={size} onChange={(e) => setSizeAnd(+e.target.value)}>
          {SIZES.map((s) => (
            <option key={s} value={s}>
              {s}px
            </option>
          ))}
        </select>
      </label>
      <span className="sep" />
      <button
        className={announcing ? "on" : ""}
        onClick={() => setAnnouncing(!announcing)}
        disabled={busy !== null}
        title="Post a thread the agents will read and react to"
      >
        Announce
      </button>
      {announcing && (
        <form
          className="god-announce"
          onSubmit={(e) => {
            e.preventDefault();
            announce();
          }}
        >
          <input
            autoFocus
            placeholder="Title the agents will see on the board"
            value={title}
            maxLength={120}
            onChange={(e) => setTitle(e.target.value)}
          />
          <input
            placeholder="Message"
            value={body}
            maxLength={600}
            onChange={(e) => setBody(e.target.value)}
          />
          <button type="submit" disabled={busy !== null || !title.trim() || !body.trim()}>
            Post as The Hand of God
          </button>
          <button type="button" className="ghost" onClick={() => setAnnouncing(false)}>
            cancel
          </button>
        </form>
      )}
      <button
        className="ghost"
        title="Set the X-God-Token for servers that lock god mode (GOD_TOKEN)"
        onClick={() => {
          const t = window.prompt(
            "God token (leave empty to clear):",
            sessionStorage.getItem("godToken") ?? "",
          );
          if (t === null) return;
          if (t) sessionStorage.setItem("godToken", t);
          else sessionStorage.removeItem("godToken");
        }}
      >
        token
      </button>
      <label className="small" title="Seconds between the starts of an agent's turns (randomised by jitter)">
        cooldown {run.config.cooldown}s
        <input
          type="range"
          min={0.1}
          max={60}
          step={0.1}
          value={run.config.cooldown}
          onChange={(e) => pace(+e.target.value)}
        />
      </label>
      <label className="small" title="Pixels each agent may paint per turn">
        px/turn
        <input
          type="number"
          min={1}
          max={1024}
          value={run.config.pixels_per_turn}
          onChange={(e) => pxPerTurn(+e.target.value)}
          style={{ width: 64 }}
        />
      </label>
      <label
        className="small"
        title="Milliseconds between an agent's successive pixels (minimum 10 ms)"
      >
        px gap {Math.round(run.config.pixel_interval * 1000)}ms
        <input
          type="range"
          min={10}
          max={1000}
          step={10}
          value={Math.round(run.config.pixel_interval * 1000)}
          onChange={(e) => pxInterval(+e.target.value)}
        />
      </label>
      {run.config.policy !== "random" && (
        <>
          <label
            className="small"
            title="Switch the model every agent and the narrator use from their next turn"
          >
            model
            <select value={modelChoice} onChange={(e) => setModelChoice(e.target.value)}>
              {models.length === 0 && <option value={run.config.model}>{run.config.model}</option>}
              {models.map((m) => (
                <option key={m.id} value={m.id} disabled={!m.available}>
                  {m.label}
                  {m.available ? "" : " — unavailable"}
                </option>
              ))}
              {models.length > 0 && !models.some((m) => m.id === modelChoice) && (
                <option value={modelChoice}>{modelChoice}</option>
              )}
            </select>
          </label>
          <button
            className="ghost"
            disabled={busy !== null || modelChoice === run.config.model}
            title="Agents and the narrator switch to this model on their next turn"
            onClick={() =>
              void act("model", () => api.godModel(runId, modelChoice).then(onRun))
            }
          >
            Switch model
          </button>
        </>
      )}
      <span className="sep" />
      <button
        className="danger"
        disabled={busy !== null}
        onClick={() =>
          window.confirm("Wipe the whole canvas?") &&
          act("clear", () => api.godClearCanvas(runId).then(onRun))
        }
      >
        Clear canvas
      </button>
      <button
        className="danger"
        disabled={busy !== null}
        onClick={() =>
          window.confirm("Delete every thread on the board?") &&
          act("clear", () => api.godClearBoard(runId).then(onRun))
        }
      >
        Clear board
      </button>
      <span className="spacer" />
      <button className="ghost" onClick={onClose}>
        done
      </button>
    </div>
  );
}

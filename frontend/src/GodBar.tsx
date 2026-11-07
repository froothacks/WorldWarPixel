import { useState } from "react";
import { api, type RunSummary } from "./api";

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
  onClose: () => void;
}

const SIZES = [1, 2, 4, 8, 16];

export default function GodBar({
  runId,
  run,
  palette,
  tool,
  onTool,
  onRun,
  onClose,
}: Props) {
  const [color, setColor] = useState(5);
  const [size, setSize] = useState(4);
  const [busy, setBusy] = useState<string | null>(null);

  const act = async (label: string, fn: () => Promise<unknown>) => {
    setBusy(label);
    try {
      await fn();
    } catch (e) {
      alert((e as Error).message);
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

  const announce = () => {
    const title = window.prompt(
      "Announcement title (agents will see this thread on the board):",
    );
    if (!title) return;
    const body = window.prompt("Message:");
    if (!body) return;
    void act("announce", () => api.godAnnounce(runId, title, body));
  };
  const pace = (cooldown: number) =>
    void act("pace", () => api.godPace(runId, { cooldown }).then(onRun));
  const pxPerTurn = (pixels_per_turn: number) =>
    void act("pace", () => api.godPace(runId, { pixels_per_turn }).then(onRun));

  return (
    <div className="godbar">
      <strong title="You are the human in the loop. Everything here is visible to the agents.">
        ⚡ God mode
      </strong>
      <span className="sep" />
      <button
        className={tool?.mode === "paint" ? "primary" : ""}
        onClick={() => pick("paint")}
        title="Paint on the canvas as The Hand of God"
      >
        Brush
      </button>
      <button
        className={tool?.mode === "erase" ? "primary" : ""}
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
        onClick={announce}
        disabled={busy !== null}
        title="Post a thread the agents will read and react to"
      >
        Announce
      </button>
      <button
        className="ghost"
        title="Set the X-God-Token for servers that lock god mode (GOD_TOKEN)"
        onClick={() => {
          const t = window.prompt(
            "God token (leave empty to clear):",
            localStorage.getItem("godToken") ?? "",
          );
          if (t === null) return;
          if (t) localStorage.setItem("godToken", t);
          else localStorage.removeItem("godToken");
        }}
      >
        🔑
      </button>
      <label className="small" title="Seconds between each agent's turns">
        cooldown {run.config.cooldown}s
        <input
          type="range"
          min={0.5}
          max={60}
          step={0.5}
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
        ✕
      </button>
    </div>
  );
}

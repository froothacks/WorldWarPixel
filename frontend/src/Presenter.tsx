import { useCallback, useEffect, useMemo, useState } from "react";
import CanvasView from "./CanvasView";
import { useDebate, useDebateFocus, useStageAudio } from "./debate";
import type { StageLine } from "./api";

interface Props {
  runId: string;
  podium: number | null;
}

export default function Presenter({ runId, podium }: Props) {
  const debate = useDebate(runId);
  const focus = useDebateFocus(debate.state);
  const state = debate.state;
  const audioFilter = useCallback(
    (line: StageLine) =>
      podium === null ||
      (line.kind === "debate" && line.podium === podium) ||
      (podium === 0 && line.kind !== "debate"),
    [podium],
  );
  const currentLine = [...debate.transcript]
    .reverse()
    .find(
      (line) =>
        line.debate_id === state?.debate_id &&
        audioFilter(line),
    ) ?? null;
  const [sound, setSound] = useState(false);
  useStageAudio(runId, debate.transcript.at(-1) ?? null, sound, audioFilter);
  const highlightMask = useMemo(() => {
    const mask = new Uint8Array(65536);
    for (const debater of state?.debaters ?? []) mask[debater.id] = 1;
    return mask.some(Boolean) ? mask : null;
  }, [state?.debaters]);
  const captionColor =
    podium === null
      ? currentLine?.color_hex ?? undefined
      : state?.debaters.find((debater) => debater.podium === podium)?.color_hex;
  const toggleSound = useCallback(() => {
    setSound((enabled) => !enabled);
  }, []);

  useEffect(() => {
    document.documentElement.classList.add("stage-page");
    return () => document.documentElement.classList.remove("stage-page");
  }, []);

  return (
    <main className={`presenter ${podium !== null ? "presenter-podium" : ""}`}>
      <header className="presenter-header">
        <span>
          <strong>WorldWarPixel</strong> <span className="slash">/</span> debate #
          {state?.debate_id ?? 0} · {state?.topic || "Waiting for the next debate"}
        </span>
        <button className={sound ? "on" : ""} onClick={toggleSound}>
          Sound {sound ? "on" : "off"}
        </button>
      </header>
      <section className="presenter-canvas">
        <CanvasView
          store={debate.store}
          version={debate.version}
          agents={debate.agentMap}
          highlightAgent={null}
          highlightMask={highlightMask}
          onPickAgent={() => undefined}
          focus={focus}
        />
      </section>
      <footer className="presenter-caption" style={{ borderColor: captionColor }}>
        {state?.phase === "idle" ? (
          <div className="presenter-copy">
            <div className="presenter-waiting">
              <strong>Waiting for the next debate</strong>
              <span>{debate.headline ?? "The narrator will report when the canvas changes."}</span>
            </div>
          </div>
        ) : currentLine ? (
          <>
            <div className="presenter-copy">
              <div className="presenter-speaker">
                {currentLine.color_hex && (
                  <span className="presenter-swatch" style={{ background: currentLine.color_hex }} />
                )}
                <strong>{currentLine.kind === "verdict" ? "Verdict" : currentLine.name}</strong>
                {currentLine.country && <span>{currentLine.country}</span>}
                {currentLine.emotion && currentLine.kind === "debate" && (
                  <span className="presenter-emotion">{currentLine.emotion}</span>
                )}
              </div>
              <div className="presenter-text">{currentLine.text}</div>
            </div>
            {state?.debaters.length === 2 && (
              <div className="presenter-votes">
                {state.debaters.map((debater) => (
                  <span className="presenter-vote" key={debater.id}>
                    <span className="presenter-swatch" style={{ background: debater.color_hex }} />
                    {debater.name}
                    <strong>{state.votes[String(debater.id)] ?? 0}</strong>
                  </span>
                ))}
              </div>
            )}
          </>
        ) : (
          <div className="presenter-copy">
            <div className="presenter-waiting">Waiting for the next line…</div>
          </div>
        )}
      </footer>
    </main>
  );
}

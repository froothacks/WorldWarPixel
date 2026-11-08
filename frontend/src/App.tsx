import { useEffect, useState } from "react";
import { api, type RunSummary } from "./api";
import StartScreen from "./StartScreen";
import LiveView from "./LiveView";
import Presenter from "./Presenter";
import Audience from "./Audience";
import Screen from "./Screen";

function runFromUrl(): string | null {
  return new URLSearchParams(location.search).get("run");
}

function setUrl(runId: string | null) {
  const url = runId ? `?run=${runId}` : location.pathname;
  history.replaceState(null, "", url);
}

export default function App() {
  const params = new URLSearchParams(location.search);
  const screen = params.get("screen");
  if (screen !== null) return <Screen society={+screen || 0} />;
  const stageId = params.get("stage");
  const audienceId = params.get("audience");
  if (audienceId) return <Audience key={audienceId} runId={audienceId} />;
  if (stageId) {
    const value = params.get("podium");
    const podium = value === null ? null : Number(value);
    return <Presenter runId={stageId} podium={podium === 0 || podium === 1 ? podium : null} />;
  }
  return <RunApp />;
}

function RunApp() {
  const [run, setRun] = useState<RunSummary | null>(null);
  const [loading, setLoading] = useState(() => runFromUrl() !== null);

  useEffect(() => {
    const id = runFromUrl();
    if (!id) return;
    api
      .getRun(id)
      .then(setRun)
      .catch(() => setUrl(null))
      .finally(() => setLoading(false));
  }, []);

  const enter = (r: RunSummary) => {
    setUrl(r.id);
    setRun(r);
  };
  const leave = () => {
    setUrl(null);
    setRun(null);
  };

  if (loading) return <p className="pad muted">Loading run…</p>;
  return run ? <LiveView key={run.id} run={run} onLeave={leave} /> : <StartScreen onStarted={enter} />;
}

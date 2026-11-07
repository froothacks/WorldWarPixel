import { useEffect, useState } from "react";
import { api, type RunSummary } from "./api";
import StartScreen from "./StartScreen";
import LiveView from "./LiveView";
import StageView from "./StageView";

function runFromUrl(): string | null {
  return new URLSearchParams(location.search).get("run");
}

function setUrl(runId: string | null) {
  const url = runId ? `?run=${runId}` : location.pathname;
  history.replaceState(null, "", url);
}

export default function App() {
  const params = new URLSearchParams(location.search);
  const stageId = params.get("stage") ?? params.get("audience");
  if (stageId) {
    const podium = params.get("podium");
    return (
      <StageView
        runId={stageId}
        mode={params.has("audience") ? "audience" : "stage"}
        podium={podium === null ? null : Number(podium)}
      />
    );
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

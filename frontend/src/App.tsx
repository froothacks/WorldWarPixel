import { useEffect, useRef, useState } from "react";
import { api, type RunSummary } from "./api";
import StartScreen from "./StartScreen";
import LiveView from "./LiveView";
import Screen from "./Screen";

function runFromUrl(): string | null {
  return new URLSearchParams(location.search).get("run");
}

export default function App() {
  const screen = new URLSearchParams(location.search).get("screen");
  if (screen !== null) return <Screen society={+screen || 0} />;
  return <Main />;
}

function Main() {
  const [run, setRun] = useState<RunSummary | null>(null);
  const [loading, setLoading] = useState(() => runFromUrl() !== null);
  const loadGeneration = useRef(0);

  useEffect(() => {
    const load = (id: string) => {
      const generation = ++loadGeneration.current;
      setLoading(true);
      api
        .getRun(id)
        .then((result) => {
          if (generation === loadGeneration.current) setRun(result);
        })
        .catch(() => {
          if (generation === loadGeneration.current) {
            history.replaceState(null, "", location.pathname);
            setRun(null);
          }
        })
        .finally(() => {
          if (generation === loadGeneration.current) setLoading(false);
        });
    };

    const popstate = () => {
      const id = runFromUrl();
      if (id) {
        load(id);
      } else {
        loadGeneration.current += 1;
        setLoading(false);
        setRun(null);
      }
    };

    window.addEventListener("popstate", popstate);
    const initial = runFromUrl();
    if (initial) {
      if (history.state?.run !== initial) {
        history.replaceState(null, "", location.pathname);
        history.pushState({ run: initial }, "", `?run=${initial}`);
      }
      load(initial);
    }

    return () => {
      loadGeneration.current += 1;
      window.removeEventListener("popstate", popstate);
    };
  }, []);

  const enter = (r: RunSummary) => {
    history.pushState({ run: r.id }, "", `?run=${r.id}`);
    setRun(r);
  };
  const leave = () => {
    if (history.state?.run) {
      history.back();
    } else {
      history.replaceState(null, "", location.pathname);
      setRun(null);
    }
  };

  if (loading) return <p className="pad muted">Loading run…</p>;
  return run ? <LiveView key={run.id} run={run} onLeave={leave} /> : <StartScreen onStarted={enter} />;
}

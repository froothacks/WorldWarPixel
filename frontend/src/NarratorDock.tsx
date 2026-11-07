import type { RunStats } from "./api";

interface Props {
  stats: RunStats;
  threads: number;
}

export default function NarratorDock({ stats, threads }: Props) {
  return (
    <footer className="narrator">
      <div className="narrator-head">
        <strong>Narrator</strong>
        <span className="muted small">updates every 60 s once the narrator agent lands</span>
      </div>
      <div className="narrator-body">
        <p className="muted">
          {stats.turns.toLocaleString()} turns, {stats.pixels.toLocaleString()} pixels and {stats.posts.toLocaleString()} posts across {threads} threads so far.
        </p>
      </div>
      <form className="narrator-ask" onSubmit={(e) => e.preventDefault()}>
        <input placeholder="Ask the narrator… (coming soon)" disabled />
        <button disabled>Ask</button>
      </form>
    </footer>
  );
}

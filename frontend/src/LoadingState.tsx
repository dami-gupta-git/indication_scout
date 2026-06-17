// Loading state shown while a run is in flight (polling). A spinner + the drug
// name, then an append-only feed of the progress events the backend emits — only
// what has actually happened, newest at the bottom, no greyed-out "forthcoming"
// items. The most recent line is the live/active one (spins); earlier lines are
// done (✓). Skeleton placeholders for the KPI band and tabs keep the page reading
// as "working" rather than frozen.

import type { ProgressEvent } from "./types";

export function LoadingState({
  drug,
  progress,
}: {
  drug: string;
  progress: ProgressEvent[];
}) {
  const lastIdx = progress.length - 1;

  return (
    <div className="loading" role="status" aria-live="polite">
      <div className="loading-head">
        <span className="spinner" aria-hidden="true" />
        <div>
          <p className="loading-title">
            Analyzing {drug ? <strong>{drug}</strong> : "…"}
          </p>
          <p className="loading-sub">
            Coordinated agents are pulling live evidence — this usually takes a few minutes.
          </p>
        </div>
      </div>

      <ul className="loading-steps loading-steps-live">
        {progress.length === 0 && (
          <li className="loading-step loading-step-active">
            <span className="loading-step-marker" aria-hidden="true">
              <span className="step-spinner" />
            </span>
            <span className="loading-step-text">Starting analysis…</span>
          </li>
        )}
        {progress.map((ev, i) => {
          const active = i === lastIdx;
          return (
            <li
              key={i}
              className={`loading-step loading-step-${active ? "active" : "done"} loading-phase-${ev.phase}`}
            >
              <span className="loading-step-marker" aria-hidden="true">
                {active ? <span className="step-spinner" /> : "✓"}
              </span>
              <span className="loading-step-text">{ev.message}</span>
            </li>
          );
        })}
      </ul>

      {/* Skeleton of the KPI band + tabs that will appear on completion. */}
      <div className="skeleton-kpis" aria-hidden="true">
        {Array.from({ length: 4 }).map((_, i) => (
          <div className="skeleton-card" key={i} />
        ))}
      </div>
      <div className="skeleton-tabs" aria-hidden="true">
        {Array.from({ length: 4 }).map((_, i) => (
          <div className="skeleton-tab" key={i} />
        ))}
      </div>
      <div className="skeleton-block" aria-hidden="true" />
      <div className="skeleton-block short" aria-hidden="true" />
    </div>
  );
}

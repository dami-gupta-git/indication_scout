// Loading state shown while a run is in flight (polling). A spinner + the
// drug name, the pipeline steps the agents work through, and skeleton
// placeholders for the KPI band and tabs so the page reads as "working" rather
// than frozen. Phase 9 streaming replaces this with a live progress matrix;
// until then (and as the SSE fallback) this is the in-flight view.

const STEPS = [
  "Surfacing candidate diseases",
  "Analysing molecular mechanism (Open Targets)",
  "Searching clinical trials (ClinicalTrials.gov)",
  "Synthesizing literature evidence (PubMed)",
  "Ranking and writing the summary",
];

export function LoadingState({ drug }: { drug: string }) {
  return (
    <div className="loading" role="status" aria-live="polite">
      <div className="loading-head">
        <span className="spinner" aria-hidden="true" />
        <div>
          <p className="loading-title">
            Analysing {drug ? <strong>{drug}</strong> : "…"}
          </p>
          <p className="loading-sub">
            Coordinated agents are pulling live evidence — this usually takes a few minutes.
          </p>
        </div>
      </div>

      <ul className="loading-steps">
        {STEPS.map((s, i) => (
          <li key={s} style={{ animationDelay: `${i * 0.25}s` }}>
            {s}
          </li>
        ))}
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

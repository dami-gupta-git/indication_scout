// Layout shell + run flow (Phase 4, T4.3-T4.6). Sidebar drives the analysis;
// the main panel shows progress / error / the result tabs. Tab *content* is
// Phase 5 — here the tabs render from the final payload at a basic level.

import { useState } from "react";
import { useAnalysis } from "./useAnalysis";
import type { SupervisorOutput } from "./types";

const TABS = ["Overview", "Mechanism", "Clinical Trials", "Literature"] as const;
type Tab = (typeof TABS)[number];

export function App() {
  const { state, run, stop } = useAnalysis();
  const [drug, setDrug] = useState("");
  const [tab, setTab] = useState<Tab>("Overview");
  const [focusDisease, setFocusDisease] = useState<string | null>(null);

  const busy = state.status === "pending" || state.status === "running";
  const result = state.data?.result ?? null;

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const name = drug.trim();
    if (name) run(name);
  };

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>IndicationScout</h1>
        <form onSubmit={submit}>
          <label htmlFor="drug">Drug</label>
          <input
            id="drug"
            value={drug}
            onChange={(e) => setDrug(e.target.value)}
            placeholder="e.g. metformin"
            disabled={busy}
          />
          <button type="submit" disabled={busy || !drug.trim()}>
            {busy ? "Analysing…" : "Analyse"}
          </button>
          {busy && (
            <button type="button" className="stop" onClick={stop}>
              Stop
            </button>
          )}
        </form>

        {result && (
          <fieldset className="focus">
            <legend>Focus disease</legend>
            <label>
              <input
                type="radio"
                name="focus"
                checked={focusDisease === null}
                onChange={() => setFocusDisease(null)}
              />
              All
            </label>
            {result.top_diseases.map((d) => (
              <label key={d}>
                <input
                  type="radio"
                  name="focus"
                  checked={focusDisease === d}
                  onChange={() => setFocusDisease(d)}
                />
                {d}
              </label>
            ))}
          </fieldset>
        )}
      </aside>

      <main className="main">
        <StatusBanner status={state.status} error={state.error} />
        {result && (
          <>
            <KpiBand result={result} />
            <nav className="tabs" role="tablist">
              {TABS.map((t) => (
                <button
                  key={t}
                  role="tab"
                  aria-selected={tab === t}
                  className={tab === t ? "active" : ""}
                  onClick={() => setTab(t)}
                >
                  {t}
                </button>
              ))}
            </nav>
            <section role="tabpanel" className="tabpanel">
              <TabContent tab={tab} result={result} focusDisease={focusDisease} />
            </section>
          </>
        )}
      </main>
    </div>
  );
}

function StatusBanner({ status, error }: { status: string; error: string | null }) {
  if (status === "idle") return <p className="hint">Enter a drug name to begin.</p>;
  if (status === "pending" || status === "running")
    return (
      <p className="banner running" role="status" aria-live="polite">
        Analysing… this may take several minutes.
      </p>
    );
  if (status === "error")
    return (
      <p className="banner error" role="alert">
        Analysis failed: {error ?? "unknown error"}
      </p>
    );
  if (status === "cancelled")
    return <p className="banner cancelled">Analysis cancelled.</p>;
  return null;
}

function KpiBand({ result }: { result: SupervisorOutput }) {
  return (
    <div className="kpis">
      <Kpi label="Drug" value={result.drug_name} />
      <Kpi label="Candidates" value={String(result.candidate_diseases.length)} />
      <Kpi label="Top diseases" value={String(result.top_diseases.length)} />
      <Kpi label="Analysed" value={String(result.disease_findings.length)} />
    </div>
  );
}

function Kpi({ label, value }: { label: string; value: string }) {
  return (
    <div className="kpi">
      <span className="kpi-value">{value}</span>
      <span className="kpi-label">{label}</span>
    </div>
  );
}

// Placeholder tab rendering — Phase 5 builds the real per-tab content.
function TabContent({
  tab,
  result,
  focusDisease,
}: {
  tab: Tab;
  result: SupervisorOutput;
  focusDisease: string | null;
}) {
  if (tab === "Overview") {
    return (
      <div>
        <p>{result.summary || "No summary."}</p>
        <ul>
          {result.disease_findings
            .filter((f) => !focusDisease || f.disease === focusDisease)
            .map((f) => (
              <li key={f.disease}>
                {f.disease} <span className="muted">({f.source})</span>
              </li>
            ))}
        </ul>
      </div>
    );
  }
  return <p className="hint">{tab} — coming in Phase 5.</p>;
}

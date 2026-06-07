// Layout shell + run flow. Sidebar drives the analysis; the main panel shows
// progress / error / the result tabs rendered from the final payload.

import { useState } from "react";
import { useAnalysis } from "./useAnalysis";
import { getReportMarkdown } from "./api";
import type { SupervisorOutput } from "./types";
import { OverviewTab } from "./tabs/OverviewTab";
import { MechanismTab } from "./tabs/MechanismTab";
import { ClinicalTrialsTab } from "./tabs/ClinicalTrialsTab";
import { LiteratureTab } from "./tabs/LiteratureTab";
import sampleOutput from "./fixtures/sample-output.json";

const TABS = ["Overview", "Mechanism", "Clinical Trials", "Literature"] as const;
type Tab = (typeof TABS)[number];

export function App() {
  const { state, run, stop, loadSample } = useAnalysis();
  const [drug, setDrug] = useState("");
  const [tab, setTab] = useState<Tab>("Overview");
  const [focusDisease, setFocusDisease] = useState<string | null>(null);

  const busy = state.status === "pending" || state.status === "running";
  const result = state.data?.result ?? null;
  // The .md report is formatted backend-side; the dev "sample" job has no
  // server-side job, so the download is only available for real finished runs.
  const canDownload =
    state.status === "done" && state.jobId !== null && state.jobId !== "sample";

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const name = drug.trim();
    if (name) run(name);
  };

  const downloadReport = async () => {
    if (!state.jobId) return;
    const md = await getReportMarkdown(state.jobId);
    const drugName = state.data?.drug_name ?? "report";
    const url = URL.createObjectURL(new Blob([md], { type: "text/markdown" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `indication_scout_${drugName.replace(/\s+/g, "_")}.md`;
    a.click();
    URL.revokeObjectURL(url);
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

        {import.meta.env.DEV && (
          <button
            type="button"
            className="dev-sample"
            onClick={() => loadSample(sampleOutput as unknown as SupervisorOutput)}
            disabled={busy}
          >
            Load sample data (dev)
          </button>
        )}

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
            {result.disease_findings.map((f) => (
              <label key={f.disease}>
                <input
                  type="radio"
                  name="focus"
                  checked={focusDisease === f.disease}
                  onChange={() => setFocusDisease(f.disease)}
                />
                {f.disease}
              </label>
            ))}
          </fieldset>
        )}

        {canDownload && (
          <button type="button" className="download" onClick={downloadReport}>
            Download report (.md)
          </button>
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
              <TabContent
                tab={tab}
                result={result}
                focusDisease={focusDisease}
                onFocus={setFocusDisease}
              />
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
  // Mirrors app.py's top-band metrics.
  const totalTrials = result.disease_findings.reduce(
    (sum, f) => sum + (f.clinical_trials?.search?.total_count ?? 0),
    0,
  );
  const totalStudies = result.disease_findings.reduce(
    (sum, f) => sum + (f.literature?.evidence_summary?.study_count ?? 0),
    0,
  );
  return (
    <div className="kpis">
      <Kpi label="Candidate diseases" value={String(result.candidate_diseases.length)} />
      <Kpi label="Investigated" value={String(result.disease_findings.length)} />
      <Kpi label="Total trials" value={String(totalTrials)} />
      <Kpi label="Total studies" value={String(totalStudies)} />
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

function TabContent({
  tab,
  result,
  focusDisease,
  onFocus,
}: {
  tab: Tab;
  result: SupervisorOutput;
  focusDisease: string | null;
  onFocus: (disease: string) => void;
}) {
  switch (tab) {
    case "Overview":
      return <OverviewTab result={result} focusDisease={focusDisease} onFocus={onFocus} />;
    case "Mechanism":
      return <MechanismTab result={result} focusDisease={focusDisease} onFocus={onFocus} />;
    case "Clinical Trials":
      return <ClinicalTrialsTab result={result} focusDisease={focusDisease} />;
    case "Literature":
      return <LiteratureTab result={result} focusDisease={focusDisease} />;
  }
}

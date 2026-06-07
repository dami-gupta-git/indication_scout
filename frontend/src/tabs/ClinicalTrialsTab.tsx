// Clinical Trials tab. Focus-disease driven: KPIs, status breakdown,
// completed/terminated/competitor tables, NCT links. Static tables + a textual
// status breakdown stand in for the bar chart for now.

import type { SupervisorOutput } from "../types";
import { NctLink } from "../components/links";
import { CompletedTrialsTable } from "../tables/CompletedTrialsTable";
import { CompetitorsTable } from "../tables/CompetitorsTable";

const TERMINATED_LIMIT = 15;

export function ClinicalTrialsTab({
  result,
  focusDisease,
}: {
  result: SupervisorOutput;
  focusDisease: string | null;
}) {
  if (!focusDisease) {
    return (
      <p className="hint">Select a focus disease in the sidebar to view its trials.</p>
    );
  }

  const finding = result.disease_findings.find((f) => f.disease === focusDisease);
  if (!finding) {
    return <p className="hint">Selected disease has no findings.</p>;
  }

  const ct = finding.clinical_trials;

  return (
    <div className="trials">
      <h3>Clinical trials — {focusDisease}</h3>
      <p className="caption">Source: {finding.source}</p>

      {ct === null ? (
        <p className="muted">No clinical trials data available.</p>
      ) : (
        <>
          {ct.summary && <p>{ct.summary}</p>}

          <div className="kpis">
            <Kpi label="Total trials" value={ct.search?.total_count ?? 0} />
            <Kpi label="Recruiting" value={ct.search?.by_status["RECRUITING"] ?? 0} />
            <Kpi
              label="Active (not recruiting)"
              value={ct.search?.by_status["ACTIVE_NOT_RECRUITING"] ?? 0}
            />
          </div>

          {ct.search && Object.keys(ct.search.by_status).length > 0 && (
            <>
              <h4>Status breakdown</h4>
              <ul className="status-breakdown">
                {Object.entries(ct.search.by_status)
                  .sort((a, b) => b[1] - a[1])
                  .map(([status, count]) => (
                    <li key={status}>
                      <span className="status-name">{status}</span>
                      <span className="status-count">{count}</span>
                    </li>
                  ))}
              </ul>
            </>
          )}

          {ct.completed && ct.completed.trials.length > 0 && (
            <>
              <h4>Completed trials ({ct.completed.total_count} total)</h4>
              <CompletedTrialsTable trials={ct.completed.trials} />
            </>
          )}

          {ct.terminated && ct.terminated.trials.length > 0 && (
            <>
              <h4>Terminated trials ({ct.terminated.total_count})</h4>
              <div className="terminated-list">
                {ct.terminated.trials.slice(0, TERMINATED_LIMIT).map((t) => (
                  <div className="card" key={t.nct_id}>
                    <div>
                      <NctLink nctId={t.nct_id} /> — {t.title || "no title"}
                    </div>
                    <p className="caption">
                      {t.phase || "Unknown phase"}
                      {t.why_stopped && <> · {t.why_stopped}</>}
                    </p>
                  </div>
                ))}
              </div>
            </>
          )}

          {ct.landscape && ct.landscape.competitors.length > 0 && (
            <>
              <h4>Competitive landscape ({ct.landscape.competitors.length})</h4>
              <CompetitorsTable competitors={ct.landscape.competitors} />
            </>
          )}
        </>
      )}
    </div>
  );
}

function Kpi({ label, value }: { label: string; value: number }) {
  return (
    <div className="kpi">
      <span className="kpi-value">{value}</span>
      <span className="kpi-label">{label}</span>
    </div>
  );
}

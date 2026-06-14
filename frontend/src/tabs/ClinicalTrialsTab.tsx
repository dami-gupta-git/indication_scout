// Clinical Trials tab. Focus-disease driven: KPIs, status donut, clickable
// phase funnel that filters the completed-trials table, terminated cards, and
// the competitor table.

import { useState } from "react";
import type { CandidateFindings, SupervisorOutput } from "../types";
import { NctLink } from "../components/links";
import { CompletedTrialsTable } from "../tables/CompletedTrialsTable";
import { CompetitorsTable } from "../tables/CompetitorsTable";
import { StatusDonut } from "../charts/StatusDonut";
import { PhaseFunnel } from "../charts/PhaseFunnel";
import { Markdown } from "../components/Markdown";
import { phaseSlices, statusSlices } from "../charts/chartData";
import { partitionTrials } from "./trialFilter";

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

  // Keyed on the disease so the funnel/table phase filter resets on switch.
  return <TrialsBody key={focusDisease} finding={finding} />;
}

function TrialsBody({ finding }: { finding: CandidateFindings }) {
  // Phase filter shared between the phase funnel and the completed-trials table.
  const [phaseFilter, setPhaseFilter] = useState<string | null>(null);

  const ct = finding.clinical_trials;

  // Hide trials the agent judged a different disease/drug from the tables, while
  // keeping the verbatim total_count headers. Mirrors the markdown report.
  const contaminated = ct?.contaminated_nct_ids ?? [];
  const completedSplit = partitionTrials(ct?.completed?.trials ?? [], contaminated);
  const terminatedSplit = partitionTrials(ct?.terminated?.trials ?? [], contaminated);
  const excludedCount = completedSplit.excluded.length + terminatedSplit.excluded.length;

  return (
    <div className="trials">
      <h3>Clinical trials — {finding.disease}</h3>
      <p className="caption">Source: {finding.source}</p>

      {ct === null ? (
        <p className="muted">No clinical trials data available.</p>
      ) : (
        <>
          {ct.summary && <Markdown>{ct.summary}</Markdown>}

          <div className="metrics">
            <Kpi label="Total trials" value={ct.search?.total_count ?? 0} />
            <Kpi label="Recruiting" value={ct.search?.by_status["RECRUITING"] ?? 0} />
            <Kpi
              label="Active (not recruiting)"
              value={ct.search?.by_status["ACTIVE_NOT_RECRUITING"] ?? 0}
            />
          </div>

          {ct.search && ct.search.total_count > 0 && (
            <>
              <h4>Status breakdown</h4>
              <StatusDonut
                slices={statusSlices(ct.search.by_status)}
                total={ct.search.total_count}
              />
            </>
          )}

          {ct.completed && completedSplit.shown.length > 0 && (
            <>
              <h4>Completed trials ({ct.completed.total_count} total)</h4>
              <PhaseFunnel
                slices={phaseSlices(completedSplit.shown)}
                active={phaseFilter}
                onSelect={setPhaseFilter}
              />
              <CompletedTrialsTable
                trials={completedSplit.shown}
                phase={phaseFilter}
                onPhaseChange={setPhaseFilter}
              />
            </>
          )}

          {ct.terminated && terminatedSplit.shown.length > 0 && (
            <>
              <h4>Terminated trials ({ct.terminated.total_count})</h4>
              <div className="terminated-list">
                {terminatedSplit.shown.slice(0, TERMINATED_LIMIT).map((t) => (
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

          {excludedCount > 0 && (
            <details className="excluded-trials">
              <summary>
                {excludedCount} trial(s) excluded as a different indication
              </summary>
              <ul>
                {[...completedSplit.excluded, ...terminatedSplit.excluded].map((t) => (
                  <li key={t.nct_id}>
                    <NctLink nctId={t.nct_id} /> — {t.title || "no title"}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </>
      )}
    </div>
  );
}

function Kpi({ label, value }: { label: string; value: number }) {
  return (
    <div className="metric">
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
    </div>
  );
}

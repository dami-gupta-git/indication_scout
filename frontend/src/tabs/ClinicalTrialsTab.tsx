// Clinical Trials tab. Focus-disease driven: KPIs, status donut, clickable
// phase funnel that filters the completed-trials table, terminated cards, and
// the competitor table.

import { useState } from "react";
import type {
  CandidateFindings,
  SupervisorOutput,
  TrialRelevanceCoverage,
} from "../types";
import { NctLink } from "../components/links";
import { CompletedTrialsTable } from "../tables/CompletedTrialsTable";
import { CompetitorsTable } from "../tables/CompetitorsTable";
import { StatusDonut } from "../charts/StatusDonut";
import { PhaseFunnel } from "../charts/PhaseFunnel";
import { Markdown } from "../components/Markdown";
import { phaseSlices, statusSlices } from "../charts/chartData";
import { partitionTrials } from "./trialFilter";

const TERMINATED_LIMIT = 15;

function coverageLabel(coverage: TrialRelevanceCoverage | null): string {
  if (coverage === null) return "relevance coverage unavailable";
  if (coverage.coverage_complete) {
    return `${coverage.relevant_records} relevant; ${coverage.contaminated_records} excluded`;
  }
  return `at least ${coverage.relevant_records} relevant among ${coverage.classified_records} reviewed; ${coverage.registry_query_matches} query matches, ${coverage.unreviewed_records} not reviewed`;
}

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

  // Only relevance-reviewed trials are evidence and appear in the tables.
  const contaminated = ct?.contaminated_nct_ids ?? [];
  const relevant = ct?.relevant_nct_ids ?? [];
  const completedSplit = partitionTrials(ct?.completed?.trials ?? [], relevant, contaminated);
  const terminatedSplit = partitionTrials(ct?.terminated?.trials ?? [], relevant, contaminated);
  const excludedCount = completedSplit.excluded.length + terminatedSplit.excluded.length;

  return (
    <div className="trials">
      <h3>Clinical trials — {finding.disease}</h3>

      {ct === null ? (
        <p className="muted">No clinical trials data available.</p>
      ) : (
        <>
          {ct.summary && <Markdown>{ct.summary}</Markdown>}

          <div className="metrics">
            <Kpi label="Relevant reviewed" value={ct.search_coverage?.relevant_records ?? "—"} />
            <Kpi
              label="Registry query matches"
              value={ct.search_coverage?.registry_query_matches ?? ct.search?.total_count ?? "—"}
            />
            <Kpi
              label="Recruiting relevant"
              value={ct.search_coverage?.relevant_by_status["RECRUITING"] ?? "—"}
            />
            <Kpi
              label="Active relevant"
              value={ct.search_coverage?.relevant_by_status["ACTIVE_NOT_RECRUITING"] ?? "—"}
            />
          </div>

          {ct.search_coverage && ct.search_coverage.relevant_records > 0 && (
            <>
              <h4>Relevant reviewed status breakdown</h4>
              <StatusDonut
                slices={statusSlices(ct.search_coverage.relevant_by_status)}
                total={ct.search_coverage.relevant_records}
              />
            </>
          )}

          {ct.completed && completedSplit.shown.length > 0 && (
            <>
              <h4>Completed trials ({coverageLabel(ct.completed_coverage)})</h4>
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
              <h4>Terminated trials ({coverageLabel(ct.terminated_coverage)})</h4>
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

function Kpi({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="metric">
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
    </div>
  );
}

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
import { phaseSlices, statusSlices } from "../charts/chartData";

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

  return (
    <div className="trials">
      <h3>Clinical trials — {finding.disease}</h3>
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

          {ct.search && ct.search.total_count > 0 && (
            <>
              <h4>Status breakdown</h4>
              <StatusDonut
                slices={statusSlices(ct.search.by_status)}
                total={ct.search.total_count}
              />
            </>
          )}

          {ct.completed && ct.completed.trials.length > 0 && (
            <>
              <h4>Completed trials ({ct.completed.total_count} total)</h4>
              <PhaseFunnel
                slices={phaseSlices(ct.completed.trials)}
                active={phaseFilter}
                onSelect={setPhaseFilter}
              />
              <CompletedTrialsTable
                trials={ct.completed.trials}
                phase={phaseFilter}
                onPhaseChange={setPhaseFilter}
              />
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

// Overview tab: narrative summary + candidate diseases with investigated
// markers. The raw summary renders as prose for now.

import type { SupervisorOutput } from "../types";

export function OverviewTab({
  result,
  focusDisease,
}: {
  result: SupervisorOutput;
  focusDisease: string | null;
}) {
  const investigated = new Set(
    result.disease_findings.map((f) => f.disease.toLowerCase().trim()),
  );

  return (
    <div className="overview">
      <h3>Summary</h3>
      {result.summary ? (
        <p className="summary">{result.summary}</p>
      ) : (
        <p className="muted">No summary produced.</p>
      )}

      <h3>Candidate diseases</h3>
      {result.candidate_diseases.length > 0 ? (
        <ul className="candidate-list">
          {result.candidate_diseases.map((c) => {
            const isInvestigated = investigated.has(c.toLowerCase().trim());
            return (
              <li key={c}>
                <strong>{c}</strong>{" "}
                {isInvestigated ? (
                  <span className="marker investigated">✓ investigated</span>
                ) : (
                  <span className="marker muted">not investigated</span>
                )}
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="muted">No candidates surfaced.</p>
      )}

      {focusDisease && (
        <p className="hint">
          Focused on <strong>{focusDisease}</strong> in the Clinical Trials and Literature tabs.
        </p>
      )}
    </div>
  );
}

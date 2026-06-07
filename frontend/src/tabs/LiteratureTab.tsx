// Literature tab. Focus-disease driven: strength/study-count KPIs, summary,
// key findings, supporting PMID links.

import type { SupervisorOutput } from "../types";
import { PmidLink } from "../components/links";

export function LiteratureTab({
  result,
  focusDisease,
}: {
  result: SupervisorOutput;
  focusDisease: string | null;
}) {
  if (!focusDisease) {
    return (
      <p className="hint">Select a focus disease in the sidebar to view its literature.</p>
    );
  }

  const finding = result.disease_findings.find((f) => f.disease === focusDisease);
  if (!finding) {
    return <p className="hint">Selected disease has no findings.</p>;
  }

  const lit = finding.literature?.evidence_summary ?? null;

  return (
    <div className="literature">
      <h3>Literature — {focusDisease}</h3>
      <p className="caption">Source: {finding.source}</p>

      {lit === null ? (
        <p className="muted">No evidence summary available.</p>
      ) : (
        <>
          <div className="metrics">
            <div className="metric">
              <span className="metric-label">Evidence strength</span>
              <span className={`metric-value strength-text strength-${lit.strength}`}>
                {lit.strength.charAt(0).toUpperCase() + lit.strength.slice(1)}
              </span>
            </div>
            <div className="metric">
              <span className="metric-label">Study count</span>
              <span className="metric-value">{lit.study_count}</span>
            </div>
          </div>

          {lit.summary && (
            <>
              <h4>Summary</h4>
              <p>{lit.summary}</p>
            </>
          )}

          {lit.key_findings.length > 0 && (
            <>
              <h4>Key findings</h4>
              <ul>
                {lit.key_findings.map((f, i) => (
                  <li key={i}>{f}</li>
                ))}
              </ul>
            </>
          )}

          {lit.supporting_pmids.length > 0 && (
            <>
              <h4>Supporting PMIDs ({lit.supporting_pmids.length})</h4>
              <p className="pmid-list">
                {lit.supporting_pmids.map((p, i) => (
                  <span key={p}>
                    {i > 0 && " · "}
                    <PmidLink pmid={p} />
                  </span>
                ))}
              </p>
            </>
          )}
        </>
      )}
    </div>
  );
}

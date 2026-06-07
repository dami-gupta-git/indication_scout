// Overview tab: narrative summary (per-candidate blurb cards), cross-disease
// comparison grid, and candidate diseases with investigated markers.

import type { SupervisorOutput } from "../types";
import { ComparisonGrid } from "../grid/ComparisonGrid";
import { SummaryBlurbCard } from "../overview/SummaryBlurbCard";
import { hasStructuredBlurb } from "../overview/blurb";
import { extractSummaryFooter } from "../overview/summaryFooter";

export function OverviewTab({
  result,
  focusDisease,
  onFocus,
}: {
  result: SupervisorOutput;
  focusDisease: string | null;
  onFocus: (disease: string) => void;
}) {
  const investigated = new Set(
    result.disease_findings.map((f) => f.disease.toLowerCase().trim()),
  );

  const findingByDisease = new Map(result.disease_findings.map((f) => [f.disease, f]));
  // Ranked blurb cards, in top_diseases order. Only genuine candidates (those
  // with structured blurb fields) get cards; prose-only demoted/approval
  // entries are excluded — they're summarized in the "Demoted —" footer.
  const rankedBlurbs = result.top_diseases
    .map((disease) => ({ disease, blurb: findingByDisease.get(disease)?.blurb }))
    .filter((x) => x.blurb != null && hasStructuredBlurb(x.blurb))
    .map((x, i) => ({ rank: i + 1, disease: x.disease, blurb: x.blurb }));
  const footer = extractSummaryFooter(result.summary);

  return (
    <div className="overview">
      <h3>Summary</h3>
      {rankedBlurbs.length > 0 ? (
        <>
          <div className="blurb-cards">
            {rankedBlurbs.map(({ rank, disease, blurb }) => (
              <SummaryBlurbCard key={disease} rank={rank} disease={disease} blurb={blurb!} />
            ))}
          </div>
          {footer && <p className="summary-footer">{footer}</p>}
        </>
      ) : result.summary ? (
        <p className="summary">{result.summary}</p>
      ) : (
        <p className="muted">No summary produced.</p>
      )}

      <h3>Disease comparison</h3>
      <p className="caption">Click a row to focus that disease in the other tabs.</p>
      <ComparisonGrid result={result} focusDisease={focusDisease} onFocus={onFocus} />

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

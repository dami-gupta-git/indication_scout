// Per-candidate summary card rendering a CandidateBlurb as a label/value field
// list plus an optional Watch line and italic prose. Mirrors the report's
// _render_blurb layout (Stage / Literature / Constraint / Active programs /
// Key risk / Assessment), shown only for non-empty fields.

import type { CandidateBlurb, EvidenceStrength } from "../types";
import { VerdictTag, StrengthBadge } from "../components/Badge";

// (field on CandidateBlurb, display label) — order matches the report formatter.
const FIELDS: [keyof CandidateBlurb, string][] = [
  ["stage", "Stage"],
  ["literature", "Literature"],
  ["blocker", "Constraint"],
  ["active_programs", "Active programs"],
  ["key_risk", "Key risk"],
];

export function SummaryBlurbCard({
  rank,
  disease,
  blurb,
  strength,
}: {
  rank: number;
  disease: string;
  blurb: CandidateBlurb;
  strength: EvidenceStrength | null;
}) {
  const rows = FIELDS.map(([key, label]) => [label, blurb[key].trim()] as const).filter(
    ([, value]) => value,
  );
  const verdict = blurb.verdict.trim();
  const watch = blurb.watch.trim();
  const prose = blurb.prose.trim();

  return (
    <div className="blurb-card">
      <h4 className="blurb-title">
        {rank}. {disease}
      </h4>
      <dl className="blurb-fields">
        {verdict && (
          <div className="blurb-row" key="Verdict">
            <dt>Verdict</dt>
            <dd>
              <VerdictTag verdict={verdict} />
            </dd>
          </div>
        )}
        {strength && (
          <div className="blurb-row" key="Evidence">
            <dt>Evidence</dt>
            <dd>
              <StrengthBadge strength={strength} />
            </dd>
          </div>
        )}
        {rows.map(([label, value]) => (
          <div className="blurb-row" key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
      {watch && (
        <p className="blurb-watch">
          <strong>Watch:</strong> {watch}
        </p>
      )}
      {prose && <p className="blurb-prose">{prose}</p>}
    </div>
  );
}

// Shared visual-encoding components. A strength badge (strong/moderate/weak/
// none) and a verdict tag, reused across the tabs. Colors keyed off the value
// so the encoding stays consistent everywhere.

import type { EvidenceStrength } from "../types";

export function StrengthBadge({ strength }: { strength: EvidenceStrength }) {
  const label = strength.charAt(0).toUpperCase() + strength.slice(1);
  return (
    <span className={`badge strength strength-${strength}`} title={`Evidence strength: ${label}`}>
      {label}
    </span>
  );
}

// Verdict is free-text from the supervisor (e.g. "Live but bottlenecked").
// No fixed enum, so a single neutral tag style.
export function VerdictTag({ verdict }: { verdict: string }) {
  if (!verdict) return null;
  return <span className="badge verdict">{verdict}</span>;
}

export function SourceTag({ source }: { source: string }) {
  return <span className={`badge source source-${source}`}>{source}</span>;
}

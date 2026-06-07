// Pure blurb predicates for the Overview summary.

import type { CandidateBlurb } from "../types";

// Structured (non-prose) fields. A genuine ranked candidate populates at least
// one; demoted/approval-relationship entries carry prose only and are NOT shown
// as cards (they live in the summary's "Demoted —" footer instead).
const STRUCTURED_FIELDS: (keyof CandidateBlurb)[] = [
  "stage",
  "literature",
  "blocker",
  "active_programs",
  "key_risk",
  "verdict",
  "watch",
];

export function hasStructuredBlurb(blurb: CandidateBlurb): boolean {
  return STRUCTURED_FIELDS.some((f) => blurb[f].trim());
}

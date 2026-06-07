// Pure data transforms for the cross-disease comparison grid. Kept separate
// from the component so the row-building and sorting logic can be unit-tested
// without a DOM. One row per investigated disease (disease_findings).

import type { EvidenceStrength, SupervisorOutput } from "../types";
import { hasStructuredBlurb } from "../overview/blurb";

export interface GridRow {
  disease: string;
  source: "competitor" | "mechanism" | "both";
  verdict: string; // "" when no blurb / un-ranked candidate
  strength: EvidenceStrength | null; // null when no literature evidence summary
  totalTrials: number | null; // null when no clinical-trials search ran
  competitors: number | null; // null when no landscape data
  recruiting: number | null; // null when no search / no RECRUITING bucket
}

export type SortKey = keyof GridRow;
export type SortDir = "asc" | "desc";

// Order strengths so sorting reflects evidence quality, not alphabetical.
const STRENGTH_RANK: Record<EvidenceStrength, number> = {
  none: 0,
  weak: 1,
  moderate: 2,
  strong: 3,
};

/**
 * Build one row per genuine candidate disease. Demoted/approval-relationship
 * entries (prose-only blurb, or no blurb) are excluded — consistent with the
 * Overview summary cards — so the grid compares real repurposing candidates.
 */
export function buildGridRows(result: SupervisorOutput): GridRow[] {
  return result.disease_findings
    .filter((f) => f.blurb != null && hasStructuredBlurb(f.blurb))
    .map((f) => {
      const search = f.clinical_trials?.search ?? null;
      return {
        disease: f.disease,
        source: f.source,
        verdict: f.blurb?.verdict ?? "",
        strength: f.literature?.evidence_summary?.strength ?? null,
        totalTrials: search ? search.total_count : null,
        competitors: f.clinical_trials?.landscape
          ? f.clinical_trials.landscape.competitors.length
          : null,
        recruiting: search ? (search.by_status["RECRUITING"] ?? 0) : null,
      };
    });
}

// Comparable scalar for a cell; null sorts last regardless of direction.
function cellValue(row: GridRow, key: SortKey): number | string | null {
  if (key === "strength") {
    return row.strength === null ? null : STRENGTH_RANK[row.strength];
  }
  return row[key];
}

/**
 * Sort rows by a column. Nulls always sink to the bottom (missing data is not
 * "smallest"). Returns a new array — does not mutate the input.
 */
export function sortGridRows(rows: GridRow[], key: SortKey, dir: SortDir): GridRow[] {
  const factor = dir === "asc" ? 1 : -1;
  return [...rows].sort((a, b) => {
    const av = cellValue(a, key);
    const bv = cellValue(b, key);
    if (av === null && bv === null) return 0;
    if (av === null) return 1; // a after b
    if (bv === null) return -1; // a before b
    if (typeof av === "number" && typeof bv === "number") {
      return (av - bv) * factor;
    }
    return String(av).localeCompare(String(bv)) * factor;
  });
}

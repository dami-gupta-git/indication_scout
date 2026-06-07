// Pure helpers turning trial data into chart slices. Kept separate from the SVG
// components so the counting/ordering logic is unit-testable.

import type { Trial } from "../types";

export interface Slice {
  label: string;
  count: number;
}

// Phase ordering for the funnel (early → late); anything unrecognized sorts last
// in its encounter order. Mirrors the registry phase labels.
const PHASE_ORDER = [
  "Early Phase 1",
  "Phase 1",
  "Phase 1/Phase 2",
  "Phase 2",
  "Phase 2/Phase 3",
  "Phase 3",
  "Phase 3/Phase 4",
  "Phase 4",
  "Not Applicable",
];

function rank(label: string): number {
  const i = PHASE_ORDER.indexOf(label);
  return i === -1 ? PHASE_ORDER.length : i;
}

/** Count trials per phase, ordered early→late. Empty phase labeled "Unknown". */
export function phaseSlices(trials: Trial[]): Slice[] {
  const counts = new Map<string, number>();
  for (const t of trials) {
    const label = t.phase || "Unknown";
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }
  return [...counts.entries()]
    .map(([label, count]) => ({ label, count }))
    .sort((a, b) => rank(a.label) - rank(b.label));
}

/** Status slices from a by_status map, dropping zero counts, largest first. */
export function statusSlices(byStatus: Record<string, number>): Slice[] {
  return Object.entries(byStatus)
    .filter(([, count]) => count > 0)
    .map(([label, count]) => ({ label, count }))
    .sort((a, b) => b.count - a.count);
}

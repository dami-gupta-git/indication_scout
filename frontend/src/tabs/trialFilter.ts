// Show only trials explicitly classified as relevant. Contaminated records are
// available for the exclusion disclosure; unclassified records remain hidden.

import type { Trial } from "../types";

export function partitionTrials(
  trials: Trial[],
  relevantNctIds: string[],
  contaminatedNctIds: string[],
): { shown: Trial[]; excluded: Trial[] } {
  const relevant = new Set(relevantNctIds);
  const contaminated = new Set(contaminatedNctIds);
  const shown: Trial[] = [];
  const excluded: Trial[] = [];
  for (const t of trials) {
    if (relevant.has(t.nct_id)) {
      shown.push(t);
    } else if (contaminated.has(t.nct_id)) {
      excluded.push(t);
    }
  }
  return { shown, excluded };
}

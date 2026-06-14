// Filter contaminated trials out of the clinical-trials tables. The agent tags
// each completed/terminated trial relevant or contaminated; contaminated NCTs
// are a different disease/drug pulled in by the recall-first search. Both the
// markdown report and this UI hide them from the tables while keeping the
// verbatim total_count header. Mirrors format_report._fmt_clinical_trials.

import type { Trial } from "../types";

export function partitionTrials(
  trials: Trial[],
  contaminatedNctIds: string[],
): { shown: Trial[]; excluded: Trial[] } {
  const contaminated = new Set(contaminatedNctIds);
  const shown: Trial[] = [];
  const excluded: Trial[] = [];
  for (const t of trials) {
    if (contaminated.has(t.nct_id)) {
      excluded.push(t);
    } else {
      shown.push(t);
    }
  }
  return { shown, excluded };
}

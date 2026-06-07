// Competitor table with sortable headers (drug/sponsor/max phase/trials) and a
// max-phase filter. Filter first, then sort, then cap the rows shown.

import { useMemo, useState } from "react";
import type { CompetitorEntry } from "../types";
import { useSort, type Accessor } from "./useSort";
import { SortHeader } from "./SortHeader";
import { FilterChips } from "./FilterChips";

const ROW_LIMIT = 25;

type Key = "drug_name" | "sponsor" | "max_phase" | "trial_count";

const ACCESSORS: Record<Key, Accessor<CompetitorEntry>> = {
  drug_name: (c) => c.drug_name,
  sponsor: (c) => c.sponsor,
  max_phase: (c) => c.max_phase,
  trial_count: (c) => c.trial_count,
};

function distinct(values: string[]): string[] {
  return [...new Set(values.filter((v) => v))];
}

export function CompetitorsTable({ competitors }: { competitors: CompetitorEntry[] }) {
  const [maxPhase, setMaxPhase] = useState<string | null>(null);

  const phaseOptions = useMemo(
    () => distinct(competitors.map((c) => c.max_phase)),
    [competitors],
  );

  const filtered = useMemo(
    () => competitors.filter((c) => maxPhase === null || c.max_phase === maxPhase),
    [competitors, maxPhase],
  );

  const { sorted, sort, toggle } = useSort<CompetitorEntry, Key>(filtered, ACCESSORS, {
    initial: { key: "trial_count", dir: "desc" },
    numericKeys: ["trial_count"],
  });
  const shown = sorted.slice(0, ROW_LIMIT);

  return (
    <div className="data-table">
      <div className="filters">
        <FilterChips
          label="Max phase"
          options={phaseOptions}
          active={maxPhase}
          onChange={setMaxPhase}
        />
      </div>
      <p className="caption">
        Showing {shown.length} of {filtered.length}
        {filtered.length !== competitors.length && ` (filtered from ${competitors.length})`}
      </p>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <SortHeader label="Drug" sortKey="drug_name" sort={sort} onToggle={toggle} />
              <SortHeader label="Sponsor" sortKey="sponsor" sort={sort} onToggle={toggle} />
              <SortHeader
                label="Max phase"
                sortKey="max_phase"
                sort={sort}
                onToggle={toggle}
              />
              <SortHeader
                label="Trials"
                sortKey="trial_count"
                sort={sort}
                onToggle={toggle}
                numeric
              />
            </tr>
          </thead>
          <tbody>
            {shown.map((c, i) => (
              <tr key={`${c.drug_name}-${c.sponsor}-${i}`}>
                <td>{c.drug_name}</td>
                <td>{c.sponsor}</td>
                <td>{c.max_phase}</td>
                <td className="num">{c.trial_count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

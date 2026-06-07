// Completed-trials table with sortable headers (title/phase/status) and phase +
// status filter chips. Filter first, then sort, then cap the rows shown.

import { useMemo, useState } from "react";
import type { Trial } from "../types";
import { NctLink } from "../components/links";
import { useSort, type Accessor } from "./useSort";
import { SortHeader } from "./SortHeader";
import { FilterChips } from "./FilterChips";

const ROW_LIMIT = 25;

// Label used for trials with an empty phase string — matches the funnel so a
// click there and the "Unknown" chip select the same rows.
const UNKNOWN_PHASE = "Unknown";

type Key = "title" | "phase" | "overall_status";

const ACCESSORS: Record<Key, Accessor<Trial>> = {
  title: (t) => t.title,
  phase: (t) => t.phase,
  overall_status: (t) => t.overall_status,
};

const phaseLabel = (t: Trial): string => t.phase || UNKNOWN_PHASE;

// Order-preserving distinct values for a chip set, skipping empties.
function distinct(values: string[]): string[] {
  return [...new Set(values.filter((v) => v))];
}

export function CompletedTrialsTable({
  trials,
  phase: phaseProp,
  onPhaseChange,
}: {
  trials: Trial[];
  // Optional controlled phase filter (driven by the phase funnel). When omitted
  // the table manages its own phase state.
  phase?: string | null;
  onPhaseChange?: (phase: string | null) => void;
}) {
  const [phaseInternal, setPhaseInternal] = useState<string | null>(null);
  const controlled = phaseProp !== undefined;
  const phase = controlled ? phaseProp : phaseInternal;
  const setPhase = controlled ? (onPhaseChange ?? (() => {})) : setPhaseInternal;
  const [status, setStatus] = useState<string | null>(null);

  const phaseOptions = useMemo(() => distinct(trials.map(phaseLabel)), [trials]);
  const statusOptions = useMemo(
    () => distinct(trials.map((t) => t.overall_status)),
    [trials],
  );

  const filtered = useMemo(
    () =>
      trials.filter(
        (t) =>
          (phase === null || phaseLabel(t) === phase) &&
          (status === null || t.overall_status === status),
      ),
    [trials, phase, status],
  );

  const { sorted, sort, toggle } = useSort<Trial, Key>(filtered, ACCESSORS);
  const shown = sorted.slice(0, ROW_LIMIT);

  return (
    <div className="data-table">
      <div className="filters">
        <FilterChips label="Phase" options={phaseOptions} active={phase} onChange={setPhase} />
        <FilterChips
          label="Status"
          options={statusOptions}
          active={status}
          onChange={setStatus}
        />
      </div>
      <p className="caption">
        Showing {shown.length} of {filtered.length}
        {filtered.length !== trials.length && ` (filtered from ${trials.length})`}
      </p>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>NCT</th>
              <SortHeader label="Title" sortKey="title" sort={sort} onToggle={toggle} />
              <SortHeader label="Phase" sortKey="phase" sort={sort} onToggle={toggle} />
              <SortHeader
                label="Status"
                sortKey="overall_status"
                sort={sort}
                onToggle={toggle}
              />
            </tr>
          </thead>
          <tbody>
            {shown.map((t) => (
              <tr key={t.nct_id}>
                <td>
                  <NctLink nctId={t.nct_id} />
                </td>
                <td>{t.title}</td>
                <td>{t.phase || "Unknown"}</td>
                <td>{t.overall_status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// Cross-disease comparison grid: one sortable row per investigated disease, with
// verdict, evidence strength, trial counts, competitors, recruiting, and source.
// Clicking a row focuses that disease across the other tabs (shared focus state).

import { useMemo, useState } from "react";
import type { SupervisorOutput } from "../types";
import { StrengthBadge, VerdictTag, SourceTag } from "../components/Badge";
import { buildGridRows, sortGridRows, type SortDir, type SortKey } from "./gridRows";

interface Column {
  key: SortKey;
  label: string;
  numeric: boolean;
}

const COLUMNS: Column[] = [
  { key: "disease", label: "Disease", numeric: false },
  { key: "verdict", label: "Verdict", numeric: false },
  { key: "strength", label: "Evidence", numeric: false },
  { key: "totalTrials", label: "Trials", numeric: true },
  { key: "competitors", label: "Competitors", numeric: true },
  { key: "recruiting", label: "Recruiting", numeric: true },
  { key: "source", label: "Source", numeric: false },
];

export function ComparisonGrid({
  result,
  focusDisease,
  onFocus,
}: {
  result: SupervisorOutput;
  focusDisease: string | null;
  onFocus: (disease: string) => void;
}) {
  const [sortKey, setSortKey] = useState<SortKey>("totalTrials");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const rows = useMemo(
    () => sortGridRows(buildGridRows(result), sortKey, sortDir),
    [result, sortKey, sortDir],
  );

  if (rows.length === 0) {
    return <p className="muted">No investigated diseases to compare.</p>;
  }

  const toggleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      // Numeric columns default to descending (highest first); text to ascending.
      const col = COLUMNS.find((c) => c.key === key);
      setSortDir(col?.numeric ? "desc" : "asc");
    }
  };

  return (
    <div className="table-scroll">
      <table className="comparison-grid">
        <thead>
          <tr>
            {COLUMNS.map((col) => {
              const active = col.key === sortKey;
              return (
                <th key={col.key} aria-sort={active ? (sortDir === "asc" ? "ascending" : "descending") : "none"}>
                  <button
                    type="button"
                    className={`sort-header${active ? " active" : ""}`}
                    onClick={() => toggleSort(col.key)}
                  >
                    {col.label}
                    {active && <span aria-hidden="true">{sortDir === "asc" ? " ▲" : " ▼"}</span>}
                  </button>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const focused = row.disease === focusDisease;
            return (
              <tr
                key={row.disease}
                className={`grid-row${focused ? " focused" : ""}`}
                aria-selected={focused}
                tabIndex={0}
                onClick={() => onFocus(row.disease)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onFocus(row.disease);
                  }
                }}
              >
                <td>{row.disease}</td>
                <td>{row.verdict ? <VerdictTag verdict={row.verdict} /> : <span className="muted">—</span>}</td>
                <td>{row.strength ? <StrengthBadge strength={row.strength} /> : <span className="muted">—</span>}</td>
                <td className="num">{row.totalTrials ?? "—"}</td>
                <td className="num">{row.competitors ?? "—"}</td>
                <td className="num">{row.recruiting ?? "—"}</td>
                <td>
                  <SourceTag source={row.source} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

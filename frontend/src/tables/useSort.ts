// Generic, reusable sort hook + pure sort helper for the data tables. Kept
// separate from any component so the comparison logic can be unit-tested.
// Nulls/empties always sink to the bottom regardless of direction.

import { useMemo, useState } from "react";

export type SortDir = "asc" | "desc";

export interface SortState<K extends string> {
  key: K | null;
  dir: SortDir;
}

// A column's sortable value extractor: number | string | null per row.
export type Accessor<T> = (row: T) => number | string | null;

/** Pure: sort a copy of `rows` by `accessor`. Nulls sink to the bottom. */
export function sortRows<T>(rows: T[], accessor: Accessor<T>, dir: SortDir): T[] {
  const factor = dir === "asc" ? 1 : -1;
  return [...rows].sort((a, b) => {
    const av = accessor(a);
    const bv = accessor(b);
    const aEmpty = av === null || av === "";
    const bEmpty = bv === null || bv === "";
    if (aEmpty && bEmpty) return 0;
    if (aEmpty) return 1;
    if (bEmpty) return -1;
    if (typeof av === "number" && typeof bv === "number") {
      return (av - bv) * factor;
    }
    return String(av).localeCompare(String(bv)) * factor;
  });
}

/**
 * Stateful sort over a fixed column set. `accessors` maps each sortable key to
 * its value extractor; `numericKeys` decides the default direction on first
 * click (numeric → desc, text → asc). Returns the sorted rows plus the current
 * state and a toggle handler for headers.
 */
export function useSort<T, K extends string>(
  rows: T[],
  accessors: Record<K, Accessor<T>>,
  opts: { initial?: SortState<K>; numericKeys?: K[] } = {},
) {
  const [sort, setSort] = useState<SortState<K>>(
    opts.initial ?? { key: null, dir: "asc" },
  );
  const numeric = new Set(opts.numericKeys ?? []);

  const sorted = useMemo(() => {
    if (sort.key === null) return rows;
    return sortRows(rows, accessors[sort.key], sort.dir);
    // accessors is a stable literal defined per-render; intentionally omitted.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows, sort]);

  const toggle = (key: K) => {
    setSort((s) =>
      s.key === key
        ? { key, dir: s.dir === "asc" ? "desc" : "asc" }
        : { key, dir: numeric.has(key) ? "desc" : "asc" },
    );
  };

  return { sorted, sort, toggle };
}

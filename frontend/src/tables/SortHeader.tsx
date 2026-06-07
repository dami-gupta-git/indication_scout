// Reusable sortable column header. Renders a <th> with a button that toggles
// sort on click and reflects the current sort via aria-sort + an arrow.

import type { SortState } from "./useSort";

export function SortHeader<K extends string>({
  label,
  sortKey,
  sort,
  onToggle,
  numeric = false,
}: {
  label: string;
  sortKey: K;
  sort: SortState<K>;
  onToggle: (key: K) => void;
  numeric?: boolean;
}) {
  const active = sort.key === sortKey;
  return (
    <th
      className={numeric ? "num" : undefined}
      aria-sort={active ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}
    >
      <button
        type="button"
        className={`sort-header${active ? " active" : ""}`}
        onClick={() => onToggle(sortKey)}
      >
        {label}
        {active && <span aria-hidden="true">{sort.dir === "asc" ? " ▲" : " ▼"}</span>}
      </button>
    </th>
  );
}

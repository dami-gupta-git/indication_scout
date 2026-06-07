// Clickable phase funnel for the completed trials. Horizontal bars sized by
// count; clicking a bar sets the active phase filter (clicking the active one
// clears it), driving the completed-trials table. Bars are real buttons for
// keyboard + a11y.

import type { Slice } from "./chartData";

export function PhaseFunnel({
  slices,
  active,
  onSelect,
}: {
  slices: Slice[];
  active: string | null;
  onSelect: (phase: string | null) => void;
}) {
  if (slices.length === 0) return null;
  const max = Math.max(...slices.map((s) => s.count));

  return (
    <div className="funnel" role="group" aria-label="Filter completed trials by phase">
      {slices.map((s) => {
        const selected = s.label === active;
        return (
          <button
            key={s.label}
            type="button"
            className={`funnel-bar${selected ? " active" : ""}`}
            aria-pressed={selected}
            onClick={() => onSelect(selected ? null : s.label)}
          >
            <span className="funnel-label">{s.label}</span>
            <span className="funnel-track">
              <span className="funnel-fill" style={{ width: `${(s.count / max) * 100}%` }} />
            </span>
            <span className="funnel-count">{s.count}</span>
          </button>
        );
      })}
    </div>
  );
}

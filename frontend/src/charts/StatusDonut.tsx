// Display-only status donut for the whole trial result set (search.by_status).
// SVG ring of arc segments + a legend. Not clickable — the completed-trials
// table it sits above is all-COMPLETED, so status slices have nothing to filter.

import type { Slice } from "./chartData";

const SIZE = 120;
const STROKE = 18;
const R = (SIZE - STROKE) / 2;
const C = 2 * Math.PI * R;

// Stable palette cycled across slices.
const COLORS = ["#2d6cdf", "#1d9b6c", "#e0a32e", "#b23a48", "#7a5bd6", "#888"];

export function StatusDonut({ slices, total }: { slices: Slice[]; total: number }) {
  if (slices.length === 0 || total === 0) return null;

  let offset = 0;
  const arcs = slices.map((s, i) => {
    const frac = s.count / total;
    const len = frac * C;
    const arc = (
      <circle
        key={s.label}
        cx={SIZE / 2}
        cy={SIZE / 2}
        r={R}
        fill="none"
        stroke={COLORS[i % COLORS.length]}
        strokeWidth={STROKE}
        strokeDasharray={`${len} ${C - len}`}
        strokeDashoffset={-offset}
      />
    );
    offset += len;
    return arc;
  });

  return (
    <div className="donut">
      <svg
        width={SIZE}
        height={SIZE}
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        role="img"
        aria-label="Trial status distribution"
      >
        {/* Rotate -90° so the first segment starts at 12 o'clock. */}
        <g transform={`rotate(-90 ${SIZE / 2} ${SIZE / 2})`}>{arcs}</g>
        <text x={SIZE / 2} y={SIZE / 2} className="donut-total" textAnchor="middle" dy="0.35em">
          {total}
        </text>
      </svg>
      <ul className="donut-legend">
        {slices.map((s, i) => (
          <li key={s.label}>
            <span className="swatch" style={{ background: COLORS[i % COLORS.length] }} />
            {s.label} <span className="muted">({s.count})</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

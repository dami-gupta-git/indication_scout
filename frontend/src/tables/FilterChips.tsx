// Reusable filter-chip row. Renders one toggle chip per distinct value plus an
// "All" reset. Single-select: clicking a chip sets the active value, clicking
// the active chip (or "All") clears the filter (null = no filter).

export function FilterChips({
  label,
  options,
  active,
  onChange,
}: {
  label: string;
  options: string[];
  active: string | null;
  onChange: (value: string | null) => void;
}) {
  if (options.length === 0) return null;
  return (
    <div className="filter-chips" role="group" aria-label={`Filter by ${label}`}>
      <span className="filter-label">{label}:</span>
      <button
        type="button"
        className={`chip${active === null ? " active" : ""}`}
        aria-pressed={active === null}
        onClick={() => onChange(null)}
      >
        All
      </button>
      {options.map((opt) => (
        <button
          key={opt}
          type="button"
          className={`chip${active === opt ? " active" : ""}`}
          aria-pressed={active === opt}
          onClick={() => onChange(active === opt ? null : opt)}
        >
          {opt}
        </button>
      ))}
    </div>
  );
}

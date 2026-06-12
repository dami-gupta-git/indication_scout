// Landing / empty-state hero shown before any analysis has run. Explains what
// the tool does. Example drugs are offered in the sidebar.

export const EXAMPLES = ["metformin", "semaglutide", "bupropion", "sildenafil"];

const STEPS = [
  {
    title: "Mechanism",
    body: "Molecular targets and disease associations from Open Targets.",
  },
  {
    title: "Clinical Trials",
    body: "Trial activity, competitive landscape, and FDA approval status from ClinicalTrials.gov.",
  },
  {
    title: "Literature",
    body: "Evidence strength synthesized from PubMed via semantic search.",
  },
];

export function LandingHero() {
  return (
    <div className="hero">
      <div className="hero-mark" aria-hidden="true">
        🔬
      </div>
      <h1 className="hero-title">Indication Scout</h1>
      <p className="hero-tagline">
        Agentic drug repurposing. Enter a drug — coordinated AI agents pull live public
        evidence from biomedical databases, then <em>synthesize and weigh</em> it for each
        candidate disease.
      </p>

      <div className="hero-steps">
        {STEPS.map((s) => (
          <div className="hero-step" key={s.title}>
            <h3>{s.title}</h3>
            <p>{s.body}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

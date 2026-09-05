import { describe, it, expect } from "vitest";
import { buildGraph } from "./graphData";
import sample from "../fixtures/sample-output.json";
import type { SupervisorOutput } from "../types";

const result = structuredClone(sample) as unknown as SupervisorOutput;
for (const finding of result.disease_findings) {
  const search = finding.clinical_trials?.search;
  if (search && finding.clinical_trials) {
    finding.clinical_trials.search_coverage = {
      registry_query_matches: search.total_count,
      retrieved_records: search.total_count,
      classified_records: search.total_count,
      relevant_records: search.total_count,
      contaminated_records: 0,
      unreviewed_records: 0,
      coverage_complete: true,
      relevant_by_status: search.by_status,
    };
  }
}

describe("buildGraph", () => {
  const g = buildGraph(result);

  it("builds the drug node, target node, and intersection disease nodes", () => {
    expect(g.nodes.find((n) => n.kind === "drug")).toMatchObject({
      id: "drug:semaglutide",
      label: "semaglutide",
    });
    // Only GLP1R, the single target that links to an investigated disease.
    expect(g.nodes.filter((n) => n.kind === "target").map((n) => n.label)).toEqual(["GLP1R"]);
    // Disease nodes = mechanism candidates ∩ investigated findings.
    expect(g.nodes.filter((n) => n.kind === "disease").map((n) => n.label).sort()).toEqual([
      "diabetes mellitus",
      "non-alcoholic fatty liver disease",
      "type 1 diabetes mellitus",
    ]);
  });

  it("draws the drug→target edge labeled by action type", () => {
    expect(
      g.edges.find((e) => e.source === "drug:semaglutide" && e.target === "target:GLP1R"),
    ).toMatchObject({ label: "AGONIST" });
  });

  it("draws target→disease edges only from mechanism candidates — no invented links", () => {
    const tdEdges = g.edges
      .filter((e) => e.source === "target:GLP1R")
      .map((e) => e.target)
      .sort();
    expect(tdEdges).toEqual([
      "disease:diabetes mellitus",
      "disease:non-alcoholic fatty liver disease",
      "disease:type 1 diabetes mellitus",
    ]);
    // parkinson disease is investigated but has NO mechanism edge → not a node.
    expect(g.nodes.some((n) => n.id === "disease:parkinson disease")).toBe(false);
    // hypertension is a mechanism candidate but NOT investigated → not a node.
    expect(g.nodes.some((n) => n.id === "disease:hypertension")).toBe(false);
  });

  it("carries trial count and source onto disease nodes", () => {
    const t1d = g.nodes.find((n) => n.id === "disease:type 1 diabetes mellitus");
    expect(t1d).toMatchObject({ trialCount: 16, source: "both" });
  });

  it("returns an empty graph when mechanism is null", () => {
    expect(buildGraph({ ...result, mechanism: null })).toEqual({ nodes: [], edges: [] });
  });
});

// Pure derivation of the drug → target → disease graph from SupervisorOutput.
// INTEGRITY: edges come ONLY from mechanism data (drug_targets + mechanism
// .candidates) — never invented. Disease nodes are the intersection of mechanism
// candidates and investigated disease_findings, so every node is both grounded
// by the mechanism AND clickable-to-focus. Kept framework-agnostic (plain node/
// edge records) so it can be unit-tested without react-flow.

import type { EvidenceStrength, SupervisorOutput } from "../types";

export type NodeKind = "drug" | "target" | "disease";

export interface GraphNode {
  id: string;
  kind: NodeKind;
  label: string;
  // disease-only extras for sizing/coloring/tooltips:
  trialCount?: number;
  source?: "competitor" | "mechanism" | "both";
  verdict?: string;
  strength?: EvidenceStrength | null;
  description?: string; // disease_description (mechanism candidate)
  targetFunction?: string; // target_function (target node)
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label: string; // action_type for drug→target; "" for target→disease
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

const nodeId = (kind: NodeKind, name: string) => `${kind}:${name}`;

/**
 * Build the graph. Returns empty graph when there is no mechanism or no target
 * has a candidate disease that was also investigated.
 */
export function buildGraph(result: SupervisorOutput): GraphData {
  const mech = result.mechanism;
  if (mech === null) return { nodes: [], edges: [] };

  const findingByDisease = new Map(result.disease_findings.map((f) => [f.disease, f]));

  // target → action_type, from the MoA entries (fallback "" if unspecified).
  const actionByTarget = new Map<string, string>();
  for (const moa of mech.mechanisms_of_action) {
    for (const sym of moa.target_symbols) {
      if (!actionByTarget.has(sym)) actionByTarget.set(sym, moa.action_type ?? "");
    }
  }

  // target → function, from candidates (first non-empty wins).
  const functionByTarget = new Map<string, string>();
  for (const c of mech.candidates) {
    if (c.target_function && !functionByTarget.has(c.target_symbol)) {
      functionByTarget.set(c.target_symbol, c.target_function);
    }
  }

  // Mechanism candidate edges, kept ONLY where the disease was investigated
  // (intersection) so every disease node is grounded AND clickable-to-focus.
  const candidateEdges = mech.candidates.filter((c) => findingByDisease.has(c.disease_name));

  const targetsWithEdges = new Set(candidateEdges.map((c) => c.target_symbol));
  const diseaseInfo = new Map<string, { description: string }>();
  for (const c of candidateEdges) {
    if (!diseaseInfo.has(c.disease_name)) {
      diseaseInfo.set(c.disease_name, { description: c.disease_description });
    }
  }

  if (targetsWithEdges.size === 0) return { nodes: [], edges: [] };

  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];

  // Drug node.
  const drugNode = nodeId("drug", result.drug_name);
  nodes.push({ id: drugNode, kind: "drug", label: result.drug_name });

  // Target nodes (only targets that actually link to an investigated disease)
  // + drug→target edges.
  for (const target of targetsWithEdges) {
    const tId = nodeId("target", target);
    nodes.push({
      id: tId,
      kind: "target",
      label: target,
      targetFunction: functionByTarget.get(target) ?? "",
    });
    edges.push({
      id: `${drugNode}->${tId}`,
      source: drugNode,
      target: tId,
      label: actionByTarget.get(target) ?? "",
    });
  }

  // Disease nodes (one per investigated candidate disease) + target→disease edges.
  for (const disease of diseaseInfo.keys()) {
    const f = findingByDisease.get(disease)!;
    const search = f.clinical_trials?.search ?? null;
    nodes.push({
      id: nodeId("disease", disease),
      kind: "disease",
      label: disease,
      trialCount: search ? search.total_count : 0,
      source: f.source,
      verdict: f.blurb?.verdict ?? "",
      strength: f.literature?.evidence_summary?.strength ?? null,
      description: diseaseInfo.get(disease)!.description,
    });
  }
  for (const c of candidateEdges) {
    edges.push({
      id: `${nodeId("target", c.target_symbol)}->${nodeId("disease", c.disease_name)}`,
      source: nodeId("target", c.target_symbol),
      target: nodeId("disease", c.disease_name),
      label: "",
    });
  }

  return { nodes, edges };
}

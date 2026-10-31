"""Render a saved MechanismGraph as a node/edge JSON graph, an HTML drawing, and a markdown explainer.

No network calls here; the HTML page loads Cytoscape.js from a CDN when opened.
"""

import html
import json
from typing import Any

from indication_scout.graph.html_template import CYTOSCAPE_URL, HTML_TEMPLATE
from indication_scout.graph.model_graph import MechanismGraph


def _evidence_text(datatype_scores: dict[str, float]) -> str:
    ordered = sorted(datatype_scores.items(), key=lambda kv: kv[1], reverse=True)
    return ", ".join(f"{name} {score:.2f}" for name, score in ordered)


def graph_to_json(graph: MechanismGraph) -> dict[str, Any]:
    """Return the graph as ``{"nodes": [...], "edges": [...]}``.

    Node kinds: drug, target, pathway, disease, therapeutic_area. Edge relations: acts_on (carries action types),
    associated_with (carries the Open Targets overall score and evidence-type scores), in_pathway, in_area. Unlinked
    candidates have no edges and are not emitted as nodes.
    """
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    drug_id = f"drug:{graph.drug_name}"
    nodes[drug_id] = {"id": drug_id, "kind": "drug", "label": graph.drug_name}

    for target in graph.targets:
        target_node = f"target:{target.target_id}"
        nodes[target_node] = {
            "id": target_node,
            "kind": "target",
            "label": target.symbol,
        }
        edges.append(
            {
                "source": drug_id,
                "target": target_node,
                "relation": "acts_on",
                "action_types": target.action_types,
            }
        )
        for pathway in target.pathways:
            pathway_node = f"pathway:{pathway.pathway_id}"
            nodes[pathway_node] = {
                "id": pathway_node,
                "kind": "pathway",
                "label": pathway.pathway_name,
                "top_level_pathway": pathway.top_level_pathway,
            }
            edges.append(
                {
                    "source": target_node,
                    "target": pathway_node,
                    "relation": "in_pathway",
                }
            )

    target_ids = {t.symbol: t.target_id for t in graph.targets}
    for cand in graph.candidates:
        if not cand.links or cand.disease_id is None:
            continue
        disease_node = f"disease:{cand.disease_id}"
        nodes[disease_node] = {
            "id": disease_node,
            "kind": "disease",
            "label": cand.disease_name,
            "candidate": cand.candidate,
        }
        for link in cand.links:
            edges.append(
                {
                    "source": f"target:{target_ids[link.target_symbol]}",
                    "target": disease_node,
                    "relation": "associated_with",
                    "overall_score": link.overall_score,
                    "datatype_scores": link.datatype_scores,
                }
            )
        for area in cand.therapeutic_areas:
            area_node = f"area:{area}"
            nodes[area_node] = {
                "id": area_node,
                "kind": "therapeutic_area",
                "label": area,
            }
            edges.append(
                {"source": disease_node, "target": area_node, "relation": "in_area"}
            )

    return {"nodes": list(nodes.values()), "edges": edges}


def graph_to_markdown(graph: MechanismGraph) -> str:
    """Render the per-candidate explanation, target pathways, target concentration, and therapeutic-area grouping."""
    action_types = {t.symbol: t.action_types for t in graph.targets}
    linked = [c for c in graph.candidates if c.links]
    unlinked = [c for c in graph.candidates if not c.links]

    lines = [f"# Mechanism graph: {graph.drug_name}", ""]
    lines.append(
        "Target–disease links are Open Targets associations. Clinical evidence can include this drug's own trials."
    )
    lines.append("")

    lines += ["## Candidates", ""]
    for i, cand in enumerate(graph.candidates, start=1):
        lines.append(f"### {i}. {cand.candidate}")
        lines.append("")
        if not cand.links:
            lines += ["No drug target linked.", ""]
            continue
        lines.append(f"Open Targets disease: {cand.disease_name} ({cand.disease_id})")
        lines.append("")
        lines.append(f"Therapeutic areas: {', '.join(cand.therapeutic_areas)}")
        lines.append("")
        lines.append("| Target | Action | OT score | Evidence types |")
        lines.append("|---|---|---|---|")
        for link in cand.links:
            score = (
                f"{link.overall_score:.2f}" if link.overall_score is not None else ""
            )
            lines.append(
                f"| {link.target_symbol} | {', '.join(action_types.get(link.target_symbol, []))} | {score} | "
                f"{_evidence_text(link.datatype_scores)} |"
            )
        lines.append("")

    lines += ["## Target pathways", ""]
    for target in graph.targets:
        lines.append(f"### {target.symbol}")
        lines.append("")
        if not target.pathways:
            lines += ["No Reactome pathways.", ""]
            continue
        by_top: dict[str, list[str]] = {}
        for pathway in target.pathways:
            by_top.setdefault(pathway.top_level_pathway, []).append(
                pathway.pathway_name
            )
        for top, names in by_top.items():
            lines.append(f"- **{top}**: {', '.join(names)}")
        lines.append("")

    lines += [
        "## Candidates per target",
        "",
        "| Target | Candidates | Count |",
        "|---|---|---|",
    ]
    for target in graph.targets:
        names = [
            c.candidate
            for c in linked
            if any(link.target_symbol == target.symbol for link in c.links)
        ]
        lines.append(f"| {target.symbol} | {', '.join(names)} | {len(names)} |")
    lines.append("")

    lines += [
        "## Candidates by therapeutic area",
        "",
        "| Therapeutic area | Candidates |",
        "|---|---|",
    ]
    by_area: dict[str, list[str]] = {}
    for cand in linked:
        for area in cand.therapeutic_areas:
            by_area.setdefault(area, []).append(cand.candidate)
    for area in sorted(by_area):
        lines.append(f"| {area} | {', '.join(by_area[area])} |")
    lines.append("")

    lines += ["## No target link", ""]
    if not unlinked:
        lines.append("None.")
    for cand in unlinked:
        reason = (
            "not resolved to an Open Targets disease"
            if cand.disease_id is None
            else f"resolved to {cand.disease_id}; no drug target associated"
        )
        lines.append(f"- {cand.candidate}: {reason}")
    lines.append("")

    return "\n".join(lines)


def graph_to_html(graph: MechanismGraph) -> str:
    """Return a self-contained HTML page drawing ``graph_to_json(graph)``.

    ``<`` in the embedded JSON is escaped so a label containing ``</script>`` cannot end the script block.
    """
    graph_json = json.dumps(graph_to_json(graph)).replace("<", "\\u003c")
    return (
        HTML_TEMPLATE.replace("__CYTOSCAPE_URL__", CYTOSCAPE_URL)
        .replace("__TITLE__", html.escape(graph.drug_name))
        .replace("__GRAPH_JSON__", graph_json)
    )

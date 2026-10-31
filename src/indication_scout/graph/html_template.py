"""Self-contained HTML page that draws a mechanism graph with Cytoscape.js.

`graph_to_html` substitutes the drug name and the node/edge JSON from `graph_to_json` into ``__TITLE__`` and
``__GRAPH_JSON__``. Layout is fixed by node kind: drug, targets (with their pathways below), candidate diseases, therapeutic
areas, left to right.
"""

CYTOSCAPE_URL = (
    "https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.30.2/cytoscape.min.js"
)

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ mechanism graph</title>
<script src="__CYTOSCAPE_URL__"></script>
<style>
  :root {
    --bg: #fafaf9; --panel: #ffffff; --text: #1c1917; --muted: #78716c; --border: #e7e5e4;
    --drug: #1d4ed8; --target: #0f766e; --pathway: #a8a29e; --disease: #b45309; --area: #7c3aed;
    --edge: #a8a29e; --edge-hi: #1c1917;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #0c0a09; --panel: #1c1917; --text: #f5f5f4; --muted: #a8a29e; --border: #292524;
      --drug: #60a5fa; --target: #2dd4bf; --pathway: #78716c; --disease: #fbbf24; --area: #c4b5fd;
      --edge: #57534e; --edge-hi: #f5f5f4;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { padding: 16px 20px 8px; }
  h1 { margin: 0 0 4px; font-size: 20px; font-weight: 600; }
  .sub { color: var(--muted); font-size: 13px; }
  .bar { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; padding: 8px 20px; }
  .legend { display: flex; flex-wrap: wrap; gap: 12px; font-size: 12px; color: var(--muted); }
  .legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 50%;
                         margin-right: 5px; vertical-align: -1px; background: var(--c); }
  label { font-size: 13px; cursor: pointer; user-select: none; }
  main { display: flex; gap: 12px; padding: 0 20px 20px; height: calc(100vh - 130px); min-height: 480px; }
  #cy { flex: 1; min-width: 0; background: var(--panel); border: 1px solid var(--border); border-radius: 8px; }
  aside { width: 320px; flex-shrink: 0; background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
          padding: 14px 16px; overflow: auto; }
  aside h2 { font-size: 15px; margin: 0 0 6px; }
  aside .kind { font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
  aside table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }
  aside td { padding: 3px 0; border-bottom: 1px solid var(--border); vertical-align: top; }
  aside td:last-child { text-align: right; font-variant-numeric: tabular-nums; }
  aside ul { margin: 6px 0 0; padding-left: 18px; font-size: 12px; }
  @media (max-width: 760px) {
    main { flex-direction: column; height: auto; }
    #cy { height: 70vh; }
    aside { width: 100%; }
  }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="sub">Drug &rarr; target &rarr; candidate disease, from Open Targets. Edge width is the association score.
    Clinical evidence can include this drug's own trials.</div>
</header>
<div class="bar">
  <div class="legend">
    <span style="--c: var(--drug)">Drug</span><span style="--c: var(--target)">Target</span>
    <span style="--c: var(--pathway)">Pathway</span><span style="--c: var(--disease)">Candidate</span>
    <span style="--c: var(--area)">Therapeutic area</span>
  </div>
  <label><input type="checkbox" id="showPathways"> Pathways</label>
  <label><input type="checkbox" id="showAreas" checked> Therapeutic areas</label>
</div>
<main>
  <div id="cy"></div>
  <aside id="detail"><div class="sub">Click a node or edge for details.</div></aside>
</main>
<script>
const GRAPH = __GRAPH_JSON__;

const css = getComputedStyle(document.documentElement);
const color = name => css.getPropertyValue("--" + name).trim();
const COLUMN = { drug: 0, target: 1, pathway: 1, disease: 2, therapeutic_area: 3 };
const COL_GAP = 300, ROW_GAP = 44;

function layoutPositions(nodes) {
  const cols = {};
  for (const n of nodes) {
    const key = n.kind === "pathway" ? "pathway" : String(COLUMN[n.kind]);
    (cols[key] = cols[key] || []).push(n);
  }
  const pos = {};
  const place = (list, x, startY) => list.forEach((n, i) => { pos[n.id] = { x, y: startY + i * ROW_GAP }; });
  for (const [key, list] of Object.entries(cols)) {
    if (key === "pathway") continue;
    place(list, Number(key) * COL_GAP, -((list.length - 1) * ROW_GAP) / 2);
  }
  const targets = cols["1"] || [];
  const pathways = cols["pathway"] || [];
  place(pathways, COL_GAP, ((targets.length - 1) * ROW_GAP) / 2 + ROW_GAP * 1.5);
  return pos;
}

const pos = layoutPositions(GRAPH.nodes);
const elements = [
  ...GRAPH.nodes.map(n => ({ group: "nodes", data: { ...n }, position: pos[n.id] })),
  ...GRAPH.edges.map((e, i) => ({ group: "edges", data: { ...e, id: "e" + i } })),
];

const cy = cytoscape({
  container: document.getElementById("cy"),
  elements,
  layout: { name: "preset" },
  wheelSensitivity: 0.2,
  style: [
    { selector: "node", style: {
        "label": "data(label)", "font-size": 11, "color": color("text"),
        "text-valign": "center", "text-halign": "right", "text-margin-x": 6,
        "width": 14, "height": 14, "background-color": color("muted") } },
    { selector: "node[kind = 'drug']", style: { "background-color": color("drug"), "width": 26, "height": 26,
        "font-size": 14, "font-weight": 600, "text-halign": "center", "text-valign": "top", "text-margin-y": -6,
        "text-margin-x": 0 } },
    { selector: "node[kind = 'target']", style: { "background-color": color("target"), "width": 18, "height": 18,
        "font-weight": 600, "text-halign": "center", "text-valign": "top", "text-margin-x": 0, "text-margin-y": -4 } },
    { selector: "node[kind = 'pathway']", style: { "background-color": color("pathway"), "width": 9, "height": 9,
        "font-size": 9, "color": color("muted") } },
    { selector: "node[kind = 'disease']", style: { "background-color": color("disease"), "width": 18, "height": 18 } },
    { selector: "node[kind = 'therapeutic_area']", style: { "background-color": color("area"), "shape": "round-rectangle",
        "width": 14, "height": 14 } },
    { selector: "edge", style: { "curve-style": "bezier", "line-color": color("edge"), "width": 1, "opacity": 0.7 } },
    { selector: "edge[relation = 'associated_with']", style: {
        "width": "mapData(overall_score, 0, 1, 1, 7)", "line-color": color("disease"), "opacity": 0.55 } },
    { selector: "edge[relation = 'acts_on']", style: { "line-color": color("target"), "width": 2 } },
    { selector: ".faded", style: { "opacity": 0.08 } },
    { selector: ".hi", style: { "opacity": 1 } },
    { selector: "edge.hi", style: { "line-color": color("edge-hi") } },
  ],
});

function applyToggles() {
  const showPathways = document.getElementById("showPathways").checked;
  const showAreas = document.getElementById("showAreas").checked;
  cy.nodes("[kind = 'pathway']").style("display", showPathways ? "element" : "none");
  cy.nodes("[kind = 'therapeutic_area']").style("display", showAreas ? "element" : "none");
  cy.fit(cy.elements(":visible"), 30);
}
document.getElementById("showPathways").addEventListener("change", applyToggles);
document.getElementById("showAreas").addEventListener("change", applyToggles);
applyToggles();

const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const KIND_LABEL = { drug: "Drug", target: "Target", pathway: "Reactome pathway", disease: "Candidate disease",
                     therapeutic_area: "Therapeutic area" };

function evidenceRows(scores) {
  return Object.entries(scores || {}).sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `<tr><td>${esc(k.replace(/_/g, " "))}</td><td>${v.toFixed(2)}</td></tr>`).join("");
}

function showNode(n) {
  const d = n.data();
  let html = `<div class="kind">${KIND_LABEL[d.kind]}</div><h2>${esc(d.label)}</h2>`;
  if (d.kind === "disease") {
    html += `<div class="sub">Ranked as &ldquo;${esc(d.candidate)}&rdquo; &middot; ${esc(d.id.replace("disease:", ""))}</div>`;
    const links = n.connectedEdges("[relation = 'associated_with']")
      .sort((a, b) => b.data("overall_score") - a.data("overall_score"));
    for (const e of links) {
      html += `<table><tr><td><b>${esc(e.source().data("label"))}</b></td>` +
              `<td><b>${e.data("overall_score").toFixed(2)}</b></td></tr>${evidenceRows(e.data("datatype_scores"))}</table>`;
    }
  } else if (d.kind === "target") {
    const acts = n.connectedEdges("[relation = 'acts_on']").map(e => (e.data("action_types") || []).join(", "));
    html += `<div class="sub">${esc(d.id.replace("target:", ""))}${acts[0] ? " &middot; " + esc(acts[0]) : ""}</div>`;
    const dis = n.connectedEdges("[relation = 'associated_with']").map(e => e.target().data("label"));
    html += `<div class="sub" style="margin-top:8px">Candidates (${dis.length})</div><ul>` +
            dis.map(x => `<li>${esc(x)}</li>`).join("") + "</ul>";
    const pw = n.connectedEdges("[relation = 'in_pathway']").map(e => e.target().data());
    if (pw.length) html += `<div class="sub" style="margin-top:8px">Pathways (${pw.length})</div><ul>` +
            pw.map(p => `<li>${esc(p.label)} <span class="sub">(${esc(p.top_level_pathway)})</span></li>`).join("") + "</ul>";
  } else if (d.kind === "pathway") {
    html += `<div class="sub">${esc(d.top_level_pathway)} &middot; ${esc(d.id.replace("pathway:", ""))}</div>`;
  } else {
    const names = n.neighborhood("node").map(x => x.data("label"));
    html += `<ul>${names.map(x => `<li>${esc(x)}</li>`).join("")}</ul>`;
  }
  document.getElementById("detail").innerHTML = html;
}

function showEdge(e) {
  const d = e.data();
  let html = `<div class="kind">${esc(d.relation.replace(/_/g, " "))}</div>` +
             `<h2>${esc(e.source().data("label"))} &rarr; ${esc(e.target().data("label"))}</h2>`;
  if (d.relation === "associated_with") {
    html += `<table><tr><td><b>Overall score</b></td><td><b>${d.overall_score.toFixed(2)}</b></td></tr>` +
            `${evidenceRows(d.datatype_scores)}</table>`;
  } else if (d.relation === "acts_on") {
    html += `<div class="sub">${esc((d.action_types || []).join(", "))}</div>`;
  }
  document.getElementById("detail").innerHTML = html;
}

function highlight(ele) {
  cy.elements().removeClass("hi").addClass("faded");
  const keep = ele.isNode() ? ele.closedNeighborhood() : ele.union(ele.connectedNodes());
  keep.removeClass("faded").addClass("hi");
}
cy.on("tap", "node", ev => { highlight(ev.target); showNode(ev.target); });
cy.on("tap", "edge", ev => { highlight(ev.target); showEdge(ev.target); });
cy.on("tap", ev => { if (ev.target === cy) { cy.elements().removeClass("faded hi"); } });
</script>
</body>
</html>
"""

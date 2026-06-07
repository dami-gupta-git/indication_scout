// drug → target → disease network graph (react-flow). Layout is three columns
// left→right (drug, targets, diseases). Disease nodes are sized by trial count
// and colored by source; clicking one focuses it across tabs. Hover shows the
// target function / disease description via the native title tooltip.

import { useMemo } from "react";
import ReactFlow, {
  Background,
  Controls,
  type Edge,
  type Node,
  MarkerType,
  Position,
} from "reactflow";
import "reactflow/dist/style.css";
import type { SupervisorOutput } from "../types";
import { buildGraph, type GraphNode } from "./graphData";

// Column x-positions and vertical spacing.
const COL_X = { drug: 0, target: 260, disease: 540 };
const ROW_GAP = 90;

const SOURCE_COLOR: Record<string, string> = {
  competitor: "#7a5bd6",
  mechanism: "#1d9b6c",
  both: "#2d6cdf",
};

// Disease node width scales modestly with trial count so size reads as "more
// activity" without dominating the layout.
function diseaseWidth(trialCount: number): number {
  return 150 + Math.min(trialCount, 100); // 150–250px
}

function toFlowNode(n: GraphNode, index: number, countInColumn: number): Node {
  // Center each column vertically around y=0.
  const y = (index - (countInColumn - 1) / 2) * ROW_GAP;

  if (n.kind === "drug") {
    return {
      id: n.id,
      position: { x: COL_X.drug, y },
      data: { label: n.label },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
      style: { background: "#1a1a1a", color: "#fff", borderRadius: 8, fontWeight: 600 },
    };
  }
  if (n.kind === "target") {
    return {
      id: n.id,
      position: { x: COL_X.target, y },
      data: {
        label: <span title={n.targetFunction || undefined}>{n.label}</span>,
      },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
      style: { background: "#eef0f3", border: "1px solid #cfd2d8", borderRadius: 8 },
    };
  }
  // disease
  const color = SOURCE_COLOR[n.source ?? "competitor"] ?? "#888";
  const diseaseLabel = `${n.label}${n.trialCount ? ` (${n.trialCount})` : ""}`;
  return {
    id: n.id,
    position: { x: COL_X.disease, y },
    data: { label: <span title={n.description || undefined}>{diseaseLabel}</span> },
    targetPosition: Position.Left,
    style: {
      width: diseaseWidth(n.trialCount ?? 0),
      background: "#fff",
      border: `2px solid ${color}`,
      borderRadius: 8,
      textTransform: "capitalize",
      cursor: "pointer",
    },
  };
}

export function MechanismGraph({
  result,
  focusDisease,
  onFocus,
}: {
  result: SupervisorOutput;
  focusDisease: string | null;
  onFocus: (disease: string) => void;
}) {
  const graph = useMemo(() => buildGraph(result), [result]);

  const { flowNodes, flowEdges } = useMemo(() => {
    // Group nodes by column for vertical centering.
    const byKind = { drug: [] as GraphNode[], target: [] as GraphNode[], disease: [] as GraphNode[] };
    for (const n of graph.nodes) byKind[n.kind].push(n);

    const flowNodes: Node[] = [];
    for (const kind of ["drug", "target", "disease"] as const) {
      byKind[kind].forEach((n, i) => {
        const node = toFlowNode(n, i, byKind[kind].length);
        if (n.kind === "disease" && `disease:${focusDisease}` === n.id) {
          node.style = { ...node.style, boxShadow: "0 0 0 3px #2d6cdf55", borderWidth: 3 };
        }
        flowNodes.push(node);
      });
    }

    const flowEdges: Edge[] = graph.edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      label: e.label || undefined,
      markerEnd: { type: MarkerType.ArrowClosed },
      style: { stroke: "#9aa0ac" },
    }));

    return { flowNodes, flowEdges };
  }, [graph, focusDisease]);

  if (graph.nodes.length === 0) {
    return <p className="muted">No mechanism-grounded connections to graph.</p>;
  }

  return (
    <div className="mech-graph" style={{ height: 420 }}>
      <ReactFlow
        nodes={flowNodes}
        edges={flowEdges}
        fitView
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable
        onNodeClick={(_, node) => {
          // Only disease nodes drive focus.
          if (node.id.startsWith("disease:")) onFocus(node.id.slice("disease:".length));
        }}
      >
        <Background color="#eee" gap={16} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}

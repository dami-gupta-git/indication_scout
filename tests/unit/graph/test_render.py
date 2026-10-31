"""Unit tests for the mechanism-graph JSON and markdown renderers."""

from indication_scout.graph.render import graph_to_json, graph_to_markdown

EXPECTED_MARKDOWN = """# Mechanism graph: drugx

Target–disease links are Open Targets associations. Clinical evidence can include this drug's own trials.

## Candidates

### 1. disease one

Open Targets disease: disease one (EFO_1)

Therapeutic areas: metabolic disease

| Target | Action | OT score | Evidence types |
|---|---|---|---|
| TB | AGONIST | 0.80 | genetic_association 0.70 |
| TA | INHIBITOR | 0.50 | clinical 0.40, literature 0.30 |

### 2. disease two

Open Targets disease: disease two official (EFO_2)

Therapeutic areas: cardiovascular disease, metabolic disease

| Target | Action | OT score | Evidence types |
|---|---|---|---|
| TB | AGONIST | 0.60 | literature 0.60, animal_model 0.20 |

### 3. disease three

No drug target linked.

## Target pathways

### TA

- **Metabolism**: Pathway one, Pathway two

### TB

- **Signal Transduction**: Pathway three

## Candidates per target

| Target | Candidates | Count |
|---|---|---|
| TA | disease one | 1 |
| TB | disease one, disease two | 2 |

## Candidates by therapeutic area

| Therapeutic area | Candidates |
|---|---|
| cardiovascular disease | disease two |
| metabolic disease | disease one, disease two |

## No target link

- disease three: not resolved to an Open Targets disease
"""


def test_graph_to_markdown(expected_graph):
    assert graph_to_markdown(expected_graph) == EXPECTED_MARKDOWN


def test_graph_to_json(expected_graph):
    assert graph_to_json(expected_graph) == {
        "nodes": [
            {"id": "drug:drugx", "kind": "drug", "label": "drugx"},
            {"id": "target:ENSG_A", "kind": "target", "label": "TA"},
            {
                "id": "pathway:R-1",
                "kind": "pathway",
                "label": "Pathway one",
                "top_level_pathway": "Metabolism",
            },
            {
                "id": "pathway:R-2",
                "kind": "pathway",
                "label": "Pathway two",
                "top_level_pathway": "Metabolism",
            },
            {"id": "target:ENSG_B", "kind": "target", "label": "TB"},
            {
                "id": "pathway:R-3",
                "kind": "pathway",
                "label": "Pathway three",
                "top_level_pathway": "Signal Transduction",
            },
            {
                "id": "disease:EFO_1",
                "kind": "disease",
                "label": "disease one",
                "candidate": "disease one",
            },
            {
                "id": "area:metabolic disease",
                "kind": "therapeutic_area",
                "label": "metabolic disease",
            },
            {
                "id": "disease:EFO_2",
                "kind": "disease",
                "label": "disease two official",
                "candidate": "disease two",
            },
            {
                "id": "area:cardiovascular disease",
                "kind": "therapeutic_area",
                "label": "cardiovascular disease",
            },
        ],
        "edges": [
            {
                "source": "drug:drugx",
                "target": "target:ENSG_A",
                "relation": "acts_on",
                "action_types": ["INHIBITOR"],
            },
            {
                "source": "target:ENSG_A",
                "target": "pathway:R-1",
                "relation": "in_pathway",
            },
            {
                "source": "target:ENSG_A",
                "target": "pathway:R-2",
                "relation": "in_pathway",
            },
            {
                "source": "drug:drugx",
                "target": "target:ENSG_B",
                "relation": "acts_on",
                "action_types": ["AGONIST"],
            },
            {
                "source": "target:ENSG_B",
                "target": "pathway:R-3",
                "relation": "in_pathway",
            },
            {
                "source": "target:ENSG_B",
                "target": "disease:EFO_1",
                "relation": "associated_with",
                "overall_score": 0.8,
                "datatype_scores": {"genetic_association": 0.7},
            },
            {
                "source": "target:ENSG_A",
                "target": "disease:EFO_1",
                "relation": "associated_with",
                "overall_score": 0.5,
                "datatype_scores": {"clinical": 0.4, "literature": 0.3},
            },
            {
                "source": "disease:EFO_1",
                "target": "area:metabolic disease",
                "relation": "in_area",
            },
            {
                "source": "target:ENSG_B",
                "target": "disease:EFO_2",
                "relation": "associated_with",
                "overall_score": 0.6,
                "datatype_scores": {"literature": 0.6, "animal_model": 0.2},
            },
            {
                "source": "disease:EFO_2",
                "target": "area:cardiovascular disease",
                "relation": "in_area",
            },
            {
                "source": "disease:EFO_2",
                "target": "area:metabolic disease",
                "relation": "in_area",
            },
        ],
    }


def test_graph_to_html_embeds_graph_json(expected_graph):
    import json

    from indication_scout.graph.html_template import CYTOSCAPE_URL
    from indication_scout.graph.render import graph_to_html

    page = graph_to_html(expected_graph)
    embedded = page.split("const GRAPH = ", 1)[1].split(";\n", 1)[0]
    assert json.loads(embedded) == graph_to_json(expected_graph)
    assert "<title>drugx mechanism graph</title>" in page
    assert f'<script src="{CYTOSCAPE_URL}"></script>' in page
    assert "__" not in page.split("<script>", 1)[1].replace("__proto__", "")

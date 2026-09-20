# opentargets-mcp

An MCP server that exposes the [Open Targets Platform](https://platform.opentargets.org) GraphQL API as six tools, for
interactive use from Claude Code and the Claude desktop app. The API is public and needs no key.

Every tool takes plain text — a gene symbol, a disease name, a drug name — resolves it to an Open Targets identifier
itself, and opens its answer with the entity it matched and the other candidates it passed over, so a wrong match is
visible rather than silent. Every list is capped, and the output states how many rows exist and how many were returned.

## Tools

| Tool | Arguments | Returns |
|---|---|---|
| `resolve` | `text`, optional `kind` | matching identifiers in one entity kind, or across targets, diseases, and drugs |
| `target_profile` | `gene` | identity, function, tractability labels, genetic constraint, safety liabilities |
| `target_diseases` | `gene`, `count` | top associated diseases with overall score, therapeutic areas, and datatypes |
| `disease_targets` | `disease`, `count` | top associated targets with overall score and datatypes |
| `known_drugs` | `name`, `kind`, `count` | drugs and clinical candidates with type, mechanism, and highest clinical stage |
| `evidence` | `gene`, `disease`, `count` | evidence rows for that pair: datatype, datasource, score, direction, literature |

Association scores run 0 to 1, aggregated by Open Targets across evidence datatypes; higher means stronger evidence.

## Install

```bash
cd opentargets-mcp
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Then register it with Claude Code from the terminal, not from inside a session:

```bash
claude mcp add --scope user opentargets -- /absolute/path/to/opentargets-mcp/.venv/bin/opentargets-mcp
```

User scope makes the server available in every project. Scope is fixed when the server is added, so changing it means
removing and re-adding. Confirm it connected:

```bash
claude mcp list
```

The desktop app takes the same command through its own configuration file. Inside a session, `/mcp` shows server status
and the tool list.

## Tests

```bash
.venv/bin/python -m pytest          # rendering helpers, no network
.venv/bin/python -m pytest -m live  # one call per tool against the real API
```

## Notes

Every GraphQL field used here was confirmed against the live API on 2026-09-19. Two shapes are worth knowing: the drug
list takes no paging argument, so the server requests the whole list and trims it; and a disease identifier may carry a
`MONDO_`, `EFO_`, or `HP_` prefix even though the API argument is named `efoId`.

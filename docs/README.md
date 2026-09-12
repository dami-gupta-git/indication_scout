# Docs

Three top-level files, then folders by purpose.

`OVERVIEW.md` is the short, concept-only description of the system. `ARCHITECTURE.md` is the long reference: layers,
agents, clients, cache, services, configuration. It points at the per-source docs rather than repeating them.
`ROADMAP.md` holds open ideas and known gaps.

| Folder | Purpose | Rule |
|---|---|---|
| `reference/` | Barebones contracts that should always match the code: data-source models and client methods, the RAG pipeline, the LLM call inventory, the report layout, the contamination glossary. | Update when the code changes. |
| `features/` | One write-up per feature as it was built: approval awareness, prompt caching, the literature and clinical-trials agents, the CI precision and recall checks, the validation-metrics plan. | Add a file when a feature lands; don't fold it into `ARCHITECTURE.md`. |
| `ops/` | Runbooks: migrations, pgvector index, observability stack, production readiness, Claude Code tooling. | |
| `performance/` | The June 2026 timing analysis and the optimizations it drove. | |
| `plans/open/`, `plans/done/` | `PLAN_*` files split by status. Gitignored. | Move a plan to `done/` when implemented or rejected. |
| `miss_analyses/` | Per-run recall post-mortems. | The open error register is `for_me/errors/errors.md`, not here. |

Findings, decisions, and personal notes live in `for_me/`, not in `docs/`.

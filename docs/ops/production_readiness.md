# IndicationScout production-readiness overview

IndicationScout is a production-deployed alpha application. Its scientific
pipeline and primary user workflow are implemented, while the runtime still
assumes a controlled demonstration or a small user group. The main production
gaps are scientific validation, durable execution, access control, and operational
monitoring.

| Area | Current state | Production gap |
|---|---|---|
| Core workflow | The application provides live biomedical retrieval, specialist agents, typed data contracts, source identifiers, caching, retries, a CLI, an API, and a React interface. | The workflow is sufficient for a controlled demonstration. |
| Deployment | The repository contains a Docker build, Railway configuration, database migrations, static frontend serving, and a health endpoint. | The health check confirms that the web process responds but does not check the database, model, cache, credentials, or upstream services. |
| Scientific reliability | Deterministic evidence gates, source-derived identifiers, structural regression tests, snapshots, and pipeline replay reduce unsupported output. | Overall correctness, judge accuracy, and run-to-run stability do not yet have completed quantitative baselines. Several scientific errors remain documented in [the error register](../for_me/errors/errors.md). |
| Testing | CI runs unit, frontend, deterministic regression, live end-to-end regression, structural, precision, and seed-recall checks. | Integration tests are not part of the default CI gate, and the live suite depends on external services and an LLM. |
| Job execution | API and CLI lifecycle state, attempts, progress events, and validated results are persisted in PostgreSQL. | Execution remains in the API or CLI process. There is no durable queue, restart recovery, expiration, or multi-instance coordination. |
| Security and cost control | Secrets are environment-driven, and production frontend requests use the same origin as the API. | No repository-level authentication, authorization, user isolation, request throttling, quotas, or protection against paid-analysis abuse is present. |
| Observability | The application emits correlated JSON logs, exposes bounded Prometheus metrics, provides a provisioned Grafana dashboard, and can send LLM traces to Langfuse. | Production metric retention, alerting, numerical objectives, incident response, and backup and restore verification remain undefined. |
| Persistence | PostgreSQL stores run, attempt, event, validated report, abstract, and embedding records. A mounted volume preserves models and cached source responses. | There is no ownership model, retention policy, or complete reproducibility record. |
| Product experience | Users can submit a drug, monitor progress, cancel a run, inspect report sections, download a report, and open supporting records. | Accounts, saved analyses, sharing, permissions, administration, and usage management are absent. |
| Demo freshness | Arbitrary drugs can run through the live pipeline. | Configured demonstration drugs may be served from committed seed reports for up to 30 days, so the displayed report is not always generated at request time. |
| Clinical use | The interface and reports state that the output is for research purposes and not for clinical use. | Clinical use would require validated performance, reproducible evidence snapshots, auditability, security review, change control, and regulatory assessment. |

The repository classifies the package as alpha in
[`pyproject.toml`](../pyproject.toml), which matches its current state.

## Classification by use

- As a public portfolio demonstration, the application is deployed and
  functionally complete.
- As an internal research application for a small, informed group, it is close to
  production use. Quantified scientific validation, persisted jobs and reports,
  and basic operational controls remain necessary.
- As a general multi-user service, it is not production-ready. Durable execution,
  access controls, usage limits, monitoring, recovery, and stronger release gates
  are still required.
- As a clinical application, it requires additional validation, provenance,
  auditability, security, and regulatory work. The research-use restriction should
  remain.

## Recommended sequence

1. Collect production baselines and define service objectives and alert thresholds.
2. Introduce a durable work queue with leases, heartbeats, and restart recovery.
3. Add authentication, authorization, rate limits, concurrency limits, and
   spending controls.
4. Add dependency-aware readiness checks, alerts, backup and restore verification,
   and an incident runbook.

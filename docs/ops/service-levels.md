# Service levels

This document defines provisional service levels for the local development environment. They
demonstrate measurement and error-budget behavior but are not production commitments. Compliance
is displayed over a rolling 24-hour window from Prometheus metrics collected from the API process.

## Proposed service-level agreement

The proposed external commitments are lower than the internal operating targets.

| Commitment | Target | Measurement |
|---|---:|---|
| API availability | 90% | The percentage of scheduled Prometheus scrapes in which the API metrics endpoint is reachable. |
| Analysis latency | 70% within 45 minutes | The percentage of successful live API analyses whose recorded duration is at most 2,700 seconds. |
| Report integrity | No failed report is published | A run marked with failed integrity status does not expose a completed report. |

Availability is sampled every 15 seconds. The latency commitment excludes CLI runs, seed reports,
cancelled analyses, and analyses that did not complete successfully. A window with no eligible
observations is reported as no data, not as a pass.

## Internal objectives

The internal objectives provide earlier warning before the external commitment is breached.

| Indicator | Objective | Error budget over 24 hours |
|---|---:|---:|
| API availability | 95% | The API may be unreachable for 5% of scheduled scrapes. |
| Analysis success | 80% | 20% of eligible live API analyses may end in error or interruption. |
| Analysis latency | 80% within 30 minutes | 20% of successful live API analyses may exceed 1,800 seconds. |
| Report integrity | 100% | No report marked as failing integrity validation may be published. |

User cancellations are excluded from the analysis-success denominator. Upstream dependency
failures are included when they cause an eligible API analysis to fail.

## Alerts

The service-level dashboard displays 24-hour compliance and error-budget consumption. Error-budget
alerts use shorter windows so a fast regression appears before the 24-hour objective is exhausted.
The API availability budget uses a one-hour burn rate; analysis success and latency use six-hour
burn rates. Each alert enters firing state after the observed burn rate remains above twice the
sustainable rate for five minutes.

Notifications are muted. Alert evaluation and alert state remain visible in Grafana.

Development failures and deliberate fault tests are included in the local measurements. The
dashboard identifies this scope at the top of the page. Production service levels require a
separate production telemetry source and a representative baseline.

## Cost baseline

Cost is measured without a budget objective. The service-level dashboard reports total measured
LLM cost, average cost per terminal attempt, average cost per candidate, and the shared overhead for
each attempt. These measurements include terminal API and CLI attempts in the selected dashboard
window. A cost objective can be defined after enough representative live runs have been collected.

## Measurement limits

Prometheus counters are process-local, although counter resets are handled by the queries. A process
failure can leave a durable run without a corresponding terminal analysis metric. CLI executions
are persisted in PostgreSQL but are outside the Prometheus-backed objectives. The integrity
rejection metric is not yet connected to an exception classification, so integrity remains an
enforced publication-state rule rather than a measured percentage.

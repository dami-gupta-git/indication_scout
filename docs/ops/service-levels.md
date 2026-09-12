# Service levels

This service-level commitment applies to the monitored IndicationScout API and browser interface
from September 11, 2026. It has no financial remedy. Compliance is measured over a rolling 30-day
window from Prometheus metrics collected from the API process.

## Service-level agreement

The active external commitment is intentionally lower than the internal operating targets.

| Commitment | Target | Measurement |
|---|---:|---|
| API availability | 99% | The percentage of scheduled Prometheus scrapes in which the API metrics endpoint is reachable. |
| Analysis latency | 90% within 15 minutes | The percentage of successful live API analyses whose recorded duration is at most 900 seconds. |
| Report integrity | No failed report is published | A run marked with failed integrity status does not expose a completed report. |

Availability is sampled every 15 seconds. The latency commitment excludes CLI runs, seed reports,
cancelled analyses, and analyses that did not complete successfully. A window with no eligible
observations is reported as no data, not as a pass.

## Internal objectives

The internal objectives provide earlier warning before the external commitment is breached.

| Indicator | Objective | Error budget over 30 days |
|---|---:|---:|
| API availability | 99.5% | The API may be unreachable for 0.5% of scheduled scrapes. |
| Analysis success | 95% | 5% of eligible live API analyses may end in error or interruption. |
| Analysis latency | 95% within 10 minutes | 5% of successful live API analyses may exceed 600 seconds. |
| Report integrity | 100% | No report marked as failing integrity validation may be published. |

User cancellations are excluded from the analysis-success denominator. Upstream dependency
failures are included when they cause an eligible API analysis to fail.

## Alerts

The service-level dashboard displays 30-day compliance and error-budget consumption. Error-budget
alerts use shorter windows so a fast regression appears before the 30-day objective is exhausted.
The API availability budget uses a one-hour burn rate; analysis success and latency use six-hour
burn rates. Each alert enters firing state after the observed burn rate remains above twice the
sustainable rate for five minutes.

Notifications are muted. Alert evaluation and alert state remain visible in Grafana.

## Measurement limits

Prometheus counters are process-local, although counter resets are handled by the queries. A process
failure can leave a durable run without a corresponding terminal analysis metric. CLI executions
are persisted in PostgreSQL but are outside the Prometheus-backed objectives. The integrity
rejection metric is not yet connected to an exception classification, so integrity remains an
enforced publication-state rule rather than a measured percentage.

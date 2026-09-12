# Production observability

IndicationScout emits correlated JSON logs and Prometheus metrics from the API process. Prometheus
collects and stores the metrics, while Grafana displays the provisioned service dashboard. The
current measurements provide operational visibility but do not define an SLA.

## Start the stack

The default Compose stack starts PostgreSQL, the application, Prometheus, and Grafana. Set a
Grafana administrator password before starting it.

```text
export GRAFANA_ADMIN_PASSWORD=<local-password>
docker compose up --build
```

The services are then available at:

| Service | Address | Purpose |
|---|---|---|
| IndicationScout | `http://localhost:8000` | Runs analyses and serves the production frontend bundle. |
| Prometheus | `http://localhost:9090` | Shows scrape status and permits direct metric queries. |
| Grafana | `http://localhost:3000` | Displays the provisioned dashboard. |

Grafana uses the username `admin` and the password supplied through
`GRAFANA_ADMIN_PASSWORD`. It loads the IndicationScout dashboards plus provisioned Prometheus and
PostgreSQL data sources at startup.

Port 3000 must be free before the stack starts. The Vite development server uses port 5173 and can
run at the same time. Use the production frontend served at `http://localhost:8000` when testing the
default Compose stack.

Run an analysis from the web interface or submit one through the API to populate the analysis
panels. Open `http://localhost:8000/metrics/` to inspect the raw Prometheus exposition, or open the
Prometheus targets page to confirm that the `indication-scout` target is being scraped. Stop the
stack with:

```text
make observability-down
```

`make observability-up` and `make observability-down` are shortcuts for the same default Compose
startup and shutdown operations.

The named Prometheus and Grafana volumes remain after this command, so local metric history and
Grafana state survive a normal stop and restart.

## Logs

Application and Uvicorn logs are written to standard output as one JSON object per line. Railway
can parse these records without a separate log collector. Every record contains a UTC timestamp,
severity, logger, service, and message. Railway environment, release, deployment, replica, and
OpenTelemetry trace identifiers are included when available.

Request logs contain a request identifier, raw client IP, inferred location when the geolocation
lookup succeeds, automation classification, route template, status, and duration. Analysis logs
add the durable run and attempt identifiers, submission source, execution mode, and progress stage
when available. Request IDs are returned in the `X-Request-ID` response header. An incoming
`X-Request-ID` is retained; otherwise, the application generates one.

Asynchronous context variables carry request, run, attempt, stage, and dependency fields into
existing logger calls. API and CLI analyses therefore use the same JSON format without passing
these identifiers through every function. External biomedical requests add a bounded dependency
name. Retried requests emit warning events with the retry number and outcome. The CLI `--verbose`
option changes its logging threshold to debug, while noisy third-party HTTP libraries remain at
warning level.

Raw IP addresses and inferred locations are retained in logs by requirement. Access to the log
backend and its retention period must be configured before production traffic is accepted. These
fields are not exported as metric labels or trace attributes.

## Telemetry destinations

Each telemetry destination has a separate operational role.

| Signal | Destination | Scope |
|---|---|---|
| Structured logs | Process standard output, viewed through Docker or Railway | HTTP requests, application lifecycle, run and attempt events, progress stages, retries, exceptions, raw IP addresses, and inferred locations. |
| OpenTelemetry traces | Langfuse, when enabled | LangChain and LLM execution, including model calls, latency, token usage, cost when available, and trace relationships. |
| Prometheus metrics | Prometheus | Aggregate request, analysis, dependency, latency, and integrity measurements with bounded labels. |
| Run measurements | PostgreSQL | Durable API and CLI attempt timing, token usage, total LLM cost, shared overhead, and candidate-level LLM cost. |
| Dashboards | Grafana | Queries Prometheus for service metrics and PostgreSQL for run and candidate detail. |
| LangSmith | None | The project does not explicitly configure a LangSmith destination. |

Langfuse tracing is opt-in. Traces are exported only when `TRACING_ENABLED=true` and the Langfuse
public key, secret key, and base URL are configured. JSON log records and Prometheus metrics are not
sent to Langfuse. Raw IP addresses, inferred locations, health traffic, database logs, and container
logs remain outside Langfuse.

When an OpenTelemetry span is active, its trace identifier is added to the JSON log record. This
allows a Railway or Docker log to be matched to the corresponding Langfuse trace. Durable run and
attempt identifiers are present in analysis logs and PostgreSQL records, but they are not currently
attached as Langfuse trace attributes. Adding those identifiers would complete direct run-to-trace
correlation without sending client IP or location data to Langfuse.

## Operating model

The observability stack separates telemetry by its access pattern and presents the results through
Grafana. This is a common production arrangement: logs, metrics, traces, and durable business
measurements do not need to share a storage backend to describe the same execution.

| System | Source of truth | Primary questions |
|---|---|---|
| Prometheus | Aggregate operational time series | Is the service available, fast enough, and within its SLO? |
| PostgreSQL | Durable run, attempt, and candidate measurements | Which run incurred the duration and cost, and how was cost distributed across candidates? |
| Langfuse, when enabled | Individual LLM generations and agent traces | Which model calls, prompts, tools, tokens, and outputs produced the result? |
| Process logs | Diagnostic events and exceptions | What happened at a specific stage, request, or dependency? |
| Grafana | No authoritative data of its own | What do the Prometheus and PostgreSQL measurements show together? |

OpenTelemetry provides the correlation convention and export path rather than replacing these
stores. Trace and span identifiers can connect logs to traces, while run and attempt identifiers
connect application events to persisted analysis records. OpenTelemetry documents this as context
propagation across traces, metrics, and logs. See the
[OpenTelemetry context propagation documentation](https://opentelemetry.io/docs/concepts/context-propagation/).

Grafana can query more than one data source in a dashboard. The current dashboard combines
Prometheus service measurements with PostgreSQL run and cost records, so separate storage does not
require separate operational views. See the
[Grafana data-source documentation](https://grafana.com/docs/learning-hub/intro-to-data-sources/00-overview/02-what-are-data-sources/).

## Langfuse and cost ownership

Langfuse is designed for LLM-specific traces, token usage, model pricing, and evaluations. It can
calculate cost for each generation from reported token usage and a matching model definition, then
aggregate cost by model, user, tag, or use case. See the
[Langfuse cost-tracking documentation](https://langfuse.com/docs/observability/features/token-and-cost-tracking/).

Langfuse is currently disabled in this project. The application therefore calculates cost from
provider-reported token usage and persists the resulting attempt and candidate totals in
PostgreSQL. Grafana reads those totals directly. This keeps the local dashboard functional without
an external LLM-observability service.

If Langfuse is enabled later, one component should own pricing calculations. Langfuse can become
the source of truth for generation-level tokens and cost, while finalized run and candidate
aggregates are copied into PostgreSQL for operational reporting. Prometheus should continue to
hold only bounded aggregate measurements. Run identifiers, candidate names, and individual LLM
calls should not become Prometheus labels because their unbounded values create high-cardinality
time series.

The intended correlation keys are:

| Scope | Correlation key |
|---|---|
| Incoming API request and its logs | `request_id` |
| Complete analysis across API or CLI | `run_id` |
| One execution or retry of an analysis | `attempt_id` |
| Logs associated with an OpenTelemetry trace | `trace_id` and `span_id` |
| Candidate-level timing and cost | `attempt_id` plus candidate name |

Direct run-to-Langfuse navigation requires `run_id` and `attempt_id` to be attached to Langfuse
trace metadata. That correlation is not implemented while Langfuse remains disabled.

## Explain the design

**Why not put everything in Langfuse?** Langfuse answers LLM-specific questions, but it is not the
service metrics backend. HTTP availability, dependency latency, active work, SLO compliance, and
infrastructure alerting remain Prometheus and Grafana responsibilities.

**Does using several stores split the measurements?** Storage is split by signal type. Shared
identifiers correlate individual executions, and Grafana combines the operational sources in one
dashboard. Detailed LLM traces can remain in Langfuse without copying high-cardinality trace data
into Prometheus.

**Why persist cost in PostgreSQL?** Per-run and per-candidate cost are durable business
measurements. They require exact identifiers and historical rows, which fit PostgreSQL better than
Prometheus labels. Grafana can query those rows alongside Prometheus measurements.

**Where does OpenTelemetry fit?** OpenTelemetry defines and transports telemetry and correlation
context. It does not require every signal to use the same backend. The current implementation uses
OpenTelemetry for traces when Langfuse is enabled; Prometheus metrics and JSON logs have separate
instrumentation paths.

**What changes when Langfuse is enabled?** Detailed model-call traces, token usage, prompt data,
and evaluations go to Langfuse. Prometheus remains the service and SLO source, PostgreSQL retains
run-level records, and Grafana remains the operational dashboard. A single pricing authority must
be selected to prevent conflicting cost totals.

## Metrics

Prometheus scrapes the API's `/metrics/` endpoint every 15 seconds and stores the resulting time
series in its named volume. Metric labels use bounded operational categories and route templates.
Drug names, disease names, raw paths, run identifiers, IP addresses, locations, URLs, and exception
messages are excluded.

| Metric group | Measurement |
|---|---|
| HTTP | Request count and duration by method, route template, and status class. |
| Analysis | Terminal outcome, duration, and currently active attempts by submission source and execution mode. |
| Dependency | Request outcome and duration by biomedical dependency and HTTP method. |
| Integrity | Reports classified as rejected before publication. |

The Prometheus values come from the running API process. CLI commands emit structured logs and
persist their run records, but their process-local metrics are not scraped by this stack. Use a web
or API analysis when demonstrating the Grafana panels.

Grafana reads Prometheus through a provisioned data source. Its service dashboard shows HTTP
request rate, HTTP latency at the 95th percentile, analysis outcomes, analysis duration at the 95th
percentile, active analyses, dependency outcomes, dependency latency, and report-integrity
rejections. The dashboard also lists the state of two provisioned Grafana alerts: an individual API
analysis active for more than two minutes, and any report-integrity rejection observed in the last
five minutes. The dashboard refreshes every 15 seconds and initially displays the preceding six
hours. The same rules are available under `Alerting` in Grafana.

The `IndicationScout service levels` dashboard shows the proposed SLA, internal SLO compliance, and
24-hour error-budget consumption for the local development environment. A notice identifies the
pre-production scope and the inclusion of development and deliberate fault-test failures. It also
shows measured LLM spend, average cost per complete successful analysis, average cost per candidate,
and tables combining duration and cost for API and CLI attempts. The definitions, scope, exclusions,
and measurement limitations are recorded in [service-levels.md](service-levels.md).

## Cost measurement

Each Anthropic response contributes its reported input, output, cache-read, and cache-write token
counts to the active attempt. Literature and clinical-trial work executed under a candidate is
assigned to that candidate. Candidate discovery, mechanism analysis, ranking, criticism, and final
report work remain shared run overhead. Shared work is not divided among candidates.

USD cost is calculated from the configured model identifier and Anthropic's published token prices.
The current table covers Claude Sonnet 4.6 and Claude Opus 4.6, including five-minute and one-hour
cache writes. If a call uses an unpriced model, its tokens are retained and the attempt is marked as
having incomplete pricing; Grafana excludes that attempt from cost aggregates rather than reporting
a partial total as complete.

Cost is persisted when an attempt completes, fails, or is cancelled. Candidate names and run IDs
remain in PostgreSQL and are not Prometheus labels. This keeps Prometheus label cardinality bounded
while allowing Grafana to display per-run and per-candidate tables. A process termination before the
terminal database transition can lose the uncommitted cost accumulated by that attempt.

The IndicationScout notification route is permanently muted. Alerts continue to be evaluated and
displayed in Grafana, but Grafana does not attempt to send email or other external notifications.

## Accepted limitations

The current implementation retains the following behavior:

- An API background analysis inherits its submission request context, so its logs can continue to
  contain the request ID, raw IP address, and inferred location.
- Geolocation runs before the health, metrics, and polling exclusions are applied. The first request
  from an uncached IP can therefore wait for the geolocation lookup.
- The integrity-rejection metric and alert exist, but current analysis failures are not classified
  as integrity failures. They remain empty until that classification is connected.
- The report-download route is not covered by the current polling-log suppression rule.

These limitations do not change report generation or persisted run results.

## Production deployment

The repository provisions the local stack through Docker Compose. Railway builds the application
Dockerfile and does not start the additional Compose services. A production Prometheus-compatible
backend must be configured to scrape or receive the Railway service metrics, and Grafana must be
connected to that backend. Production retention, authentication, alert destinations, and dashboard
access are deployment settings and are not supplied by the local Compose stack.

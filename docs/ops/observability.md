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
`GRAFANA_ADMIN_PASSWORD`. It loads the `IndicationScout service overview` dashboard and the
Prometheus data source at startup.

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
| OpenTelemetry traces | Langfuse | LangChain and LLM execution, including model calls, latency, token usage, cost when available, and trace relationships. |
| Prometheus metrics | Prometheus | Aggregate request, analysis, dependency, latency, and integrity measurements with bounded labels. |
| Dashboards | Grafana | Queries and displays the Prometheus time series. |
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
analysis active for more than ten minutes, and any report-integrity rejection observed in the last
five minutes. The dashboard refreshes every 15 seconds and initially displays the preceding six
hours. The same rules are available under `Alerting` in Grafana.

The alert rules are evaluated and displayed without an email destination. Email delivery requires
an explicit recipient and Grafana SMTP configuration; neither is stored in this repository.

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

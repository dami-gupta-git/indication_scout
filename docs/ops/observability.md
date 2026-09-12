# Production observability

IndicationScout emits correlated JSON logs and Prometheus metrics from the API process. A local
Prometheus and Grafana stack is provisioned from files in the repository. This setup establishes
the measurements needed to choose service objectives after representative production data has
been collected.

## Start the stack

The observability compose overlay starts the application, PostgreSQL, Prometheus, and Grafana.
Set a Grafana administrator password before starting it.

```text
export GRAFANA_ADMIN_PASSWORD=<local-password>
make observability-up
```

The application is available on port 8000, Prometheus on port 9090, and Grafana on port 3000.
Grafana loads the `IndicationScout service overview` dashboard and its Prometheus data source at
startup. Stop the stack with `make observability-down`.

## Logs

Application and Uvicorn logs are written as one JSON object per line. Every record contains a UTC
timestamp, severity, logger, service, and message. Railway environment, release, deployment,
replica, and OpenTelemetry trace identifiers are included when available.

Request logs contain a request identifier, raw client IP, inferred location when the geolocation
lookup succeeds, automation classification, route template, status, and duration. Analysis logs
add the durable run and attempt identifiers, submission source, and execution mode. Request IDs
are returned in the `X-Request-ID` response header.

Raw IP addresses and inferred locations are retained in logs by requirement. Access to the log
backend and its retention period must be configured before production traffic is accepted. These
fields are not exported as metric labels or trace attributes.

## Metrics

Prometheus scrapes the API's `/metrics/` endpoint every 15 seconds. Metric labels contain bounded
operational categories and route templates. Drug names, disease names, raw paths, run identifiers,
IP addresses, locations, URLs, and exception messages are excluded.

The dashboard shows HTTP request rate and latency, analysis outcomes and duration, active analyses,
dependency outcomes and latency, and report-integrity rejections. The current values are
measurements rather than SLAs. Numerical availability, completion, and latency objectives will be
set after a representative baseline is collected.

## Production deployment

The repository provisions the local stack. A production Prometheus-compatible backend must be
configured to scrape or receive the Railway service metrics, and Grafana must be connected to that
backend. Production retention, authentication, alert destinations, and dashboard access are
deployment settings and are not supplied by the local compose overlay.

"""Standalone OpenTelemetry "hello span" — proves the SDK → OTLP → Langfuse pipeline.

Run:  python scripts/otel_hello.py
Then open Langfuse → project `indication_scout` → Tracing; you'll see a trace "hello-otel".

Throwaway learning scaffolding — does NOT touch the app. No LangChain / OpenInference yet;
just the raw OTel core so each piece is visible in isolation:
  TracerProvider → BatchSpanProcessor → OTLPSpanExporter → (OTLP/HTTP) → Langfuse.

Debugging tip learned the hard way: a 200 from the OTLP endpoint does NOT mean the trace is
visible. Ingestion is async and the UI is account/project/time-filter scoped. To verify a
trace really landed, query Langfuse's READ API:
  GET {base}/api/public/traces   (Basic auth, same keys) — that is the source of truth.

Secrets are read from the environment only. Requires in .env:
  LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL
"""

import base64
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("otel_hello")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def main() -> None:
    public_key = os.environ["LANGFUSE_PUBLIC_KEY"]
    secret_key = os.environ["LANGFUSE_SECRET_KEY"]
    base_url = os.environ["LANGFUSE_BASE_URL"].rstrip("/")

    # Langfuse authenticates the OTLP endpoint with HTTP Basic auth:
    # token = base64("<public_key>:<secret_key>"), sent in the Authorization header.
    auth_token = base64.b64encode(
        f"{public_key}:{secret_key}".encode("utf-8")
    ).decode("ascii")
    endpoint = f"{base_url}/api/public/otel/v1/traces"

    # Resource = "who is emitting these spans" — surfaces as the service in the UI.
    resource = Resource.create({"service.name": "indication-scout-hello"})

    # Provider = the global factory for tracers; wired to our exporter.
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(
        endpoint=endpoint,
        headers={"Authorization": f"Basic {auth_token}"},
    )
    # BatchSpanProcessor buffers + flushes in the background (production choice). Short-lived
    # scripts MUST call provider.shutdown() to force the final flush before exit.
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    logger.info("Emitting one span to %s", endpoint)
    tracer = trace.get_tracer("otel_hello")
    with tracer.start_as_current_span("hello-otel") as span:
        span.set_attribute("learning.step", 1)
        logger.info("Inside the span — doing pretend work.")

    provider.shutdown()
    logger.info("Done. Check Langfuse → project indication_scout → Tracing for 'hello-otel'.")


if __name__ == "__main__":
    main()

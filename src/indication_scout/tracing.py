"""OpenTelemetry tracing → Langfuse for the LangChain/LangGraph agents.

Opt-in instrumentation. When `tracing_enabled` is true and Langfuse keys are present,
`setup_tracing()` wires the OTel pipeline and turns on OpenInference auto-instrumentation
so every LangChain LLM call emits a span (model, token usage, cost, latency) with NO
changes to agent/call-site code. `shutdown_tracing()` flushes spans before the process
exits.

Design rules (per project policy — instrumentation must never break a run):
- If tracing is disabled OR keys are missing, every function is a safe no-op.
- No exception from this module should propagate into the pipeline; failures are logged.

Pipeline:  Resource -> TracerProvider -> BatchSpanProcessor -> OTLPSpanExporter -> Langfuse.
Auth:      HTTP Basic, token = base64("<public_key>:<secret_key>").
Endpoint:  <langfuse_base_url>/api/public/otel/v1/traces.

To verify traces landed, query Langfuse's READ API (GET /api/public/traces) — a 200 from
the OTLP endpoint does NOT guarantee UI visibility (ingestion is async, view is scoped).
"""

import base64
import logging
import subprocess

from indication_scout.config import get_settings

logger = logging.getLogger(__name__)


def _git_commit_message() -> str | None:
    """Subject line of the current HEAD commit, or None if unavailable (not a repo,
    git missing). Operational metadata only — absence is fine, never fabricated."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--pretty=%s"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    message = result.stdout.strip()
    return message if result.returncode == 0 and message else None


# Module-level handle to the provider so shutdown_tracing() can flush it. None until
# setup_tracing() successfully initialises (and stays None when tracing is off).
_provider = None
_OTEL_TRACES_PATH = "/api/public/otel/v1/traces"
_SERVICE_NAME = "indication-scout"


def setup_tracing() -> None:
    """Initialise OTel + OpenInference if enabled and configured. Idempotent and safe.

    No-op (logs and returns) when tracing is disabled or the Langfuse keys are missing.
    Any failure is logged, not raised — a tracing problem must never break a pipeline run.
    """
    global _provider

    settings = get_settings()
    if not settings.tracing_enabled:
        logger.info("Tracing disabled (tracing_enabled=False); skipping OTel setup.")
        return
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning(
            "tracing_enabled=True but Langfuse keys are missing; tracing not started."
        )
        return
    if _provider is not None:
        logger.debug("Tracing already initialised; skipping re-init.")
        return

    try:
        # Imports are local so the optional `[tracing]` deps are only required when
        # tracing is actually turned on.
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        base_url = settings.langfuse_base_url.rstrip("/")
        endpoint = f"{base_url}{_OTEL_TRACES_PATH}"
        auth_token = base64.b64encode(
            f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode(
                "utf-8"
            )
        ).decode("ascii")

        # langfuse.release maps to the trace's `release` field; include the HEAD
        # commit message when available so each run is traceable to a checkout.
        resource_attrs = {"service.name": _SERVICE_NAME}
        commit_message = _git_commit_message()
        if commit_message is not None:
            resource_attrs["langfuse.release"] = commit_message
        provider = TracerProvider(resource=Resource.create(resource_attrs))
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(
                    endpoint=endpoint,
                    headers={"Authorization": f"Basic {auth_token}"},
                )
            )
        )
        trace.set_tracer_provider(provider)

        # The one line that makes every LangChain LLM call auto-emit a span.
        LangChainInstrumentor().instrument(tracer_provider=provider)

        _provider = provider
        logger.info("Tracing active -> %s (service=%s)", endpoint, _SERVICE_NAME)
    except Exception:
        # Never let an instrumentation problem break the run.
        logger.exception("Failed to initialise tracing; continuing without it.")
        _provider = None


def shutdown_tracing() -> None:
    """Flush and shut down the tracer provider so buffered spans are sent before exit.

    Safe no-op if tracing was never started. Errors are logged, not raised.
    """
    global _provider
    if _provider is None:
        return
    try:
        _provider.shutdown()
        logger.debug("Tracing shut down; spans flushed.")
    except Exception:
        logger.exception("Error during tracing shutdown.")
    finally:
        _provider = None

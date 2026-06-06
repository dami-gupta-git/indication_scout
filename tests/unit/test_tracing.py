"""Unit tests for indication_scout.tracing.

These test the GATING LOGIC only — no network, no real OTel export. The point of
tracing.py is that it (a) no-ops when disabled, (b) no-ops + warns when keys are missing,
(c) initialises + instruments when enabled, and (d) never raises into the pipeline. We
patch get_settings to drive each state and patch the OTel/OpenInference symbols (imported
locally inside setup_tracing) so nothing real is constructed.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import indication_scout.tracing as tracing


def _settings(**overrides):
    """Build a stand-in settings object with the tracing fields setup_tracing reads."""
    base = {
        "tracing_enabled": False,
        "langfuse_public_key": "",
        "langfuse_secret_key": "",
        "langfuse_base_url": "https://us.cloud.langfuse.com",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _reset_provider():
    """Each test starts and ends with no active provider (module-level global)."""
    tracing._provider = None
    yield
    tracing._provider = None


def test_setup_noop_when_disabled():
    """tracing_enabled=False -> no provider set, instrumentor never called."""
    with (
        patch.object(tracing, "get_settings", return_value=_settings(tracing_enabled=False)),
        patch(
            "openinference.instrumentation.langchain.LangChainInstrumentor"
        ) as instrumentor,
    ):
        tracing.setup_tracing()

    assert tracing._provider is None
    instrumentor.assert_not_called()


def test_setup_noop_when_keys_missing():
    """Enabled but blank keys -> no provider, instrumentor never called, no raise."""
    with (
        patch.object(
            tracing,
            "get_settings",
            return_value=_settings(
                tracing_enabled=True, langfuse_public_key="", langfuse_secret_key=""
            ),
        ),
        patch(
            "openinference.instrumentation.langchain.LangChainInstrumentor"
        ) as instrumentor,
    ):
        tracing.setup_tracing()

    assert tracing._provider is None
    instrumentor.assert_not_called()


def test_setup_initialises_when_enabled_and_keyed():
    """Enabled + keys -> provider set, BatchSpanProcessor added, instrumentor called once."""
    fake_provider = MagicMock(name="TracerProvider")
    with (
        patch.object(
            tracing,
            "get_settings",
            return_value=_settings(
                tracing_enabled=True,
                langfuse_public_key="pk-lf-abc",
                langfuse_secret_key="sk-lf-xyz",
                langfuse_base_url="https://us.cloud.langfuse.com",
            ),
        ),
        patch(
            "opentelemetry.sdk.trace.TracerProvider", return_value=fake_provider
        ),
        patch("opentelemetry.sdk.trace.export.BatchSpanProcessor") as bsp,
        patch(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter"
        ) as exporter,
        patch("opentelemetry.trace.set_tracer_provider") as set_provider,
        patch(
            "openinference.instrumentation.langchain.LangChainInstrumentor"
        ) as instrumentor,
    ):
        tracing.setup_tracing()

        # Exporter aimed at the correct OTLP endpoint with a Basic auth header.
        _, kwargs = exporter.call_args
        assert (
            kwargs["endpoint"]
            == "https://us.cloud.langfuse.com/api/public/otel/v1/traces"
        )
        assert kwargs["headers"]["Authorization"].startswith("Basic ")

        fake_provider.add_span_processor.assert_called_once_with(bsp.return_value)
        set_provider.assert_called_once_with(fake_provider)
        instrumentor.return_value.instrument.assert_called_once_with(
            tracer_provider=fake_provider
        )

    assert tracing._provider is fake_provider


def test_setup_swallows_errors_and_leaves_provider_none():
    """A failure inside setup must be caught (never raised) and leave _provider None."""
    with (
        patch.object(
            tracing,
            "get_settings",
            return_value=_settings(
                tracing_enabled=True,
                langfuse_public_key="pk-lf-abc",
                langfuse_secret_key="sk-lf-xyz",
            ),
        ),
        patch(
            "opentelemetry.sdk.trace.TracerProvider",
            side_effect=RuntimeError("boom"),
        ),
    ):
        tracing.setup_tracing()  # must not raise

    assert tracing._provider is None


def test_shutdown_noop_when_never_started():
    """shutdown is a safe no-op when no provider was initialised."""
    tracing._provider = None
    tracing.shutdown_tracing()  # must not raise
    assert tracing._provider is None


def test_shutdown_flushes_and_clears_provider():
    """shutdown calls provider.shutdown() and clears the module global."""
    fake_provider = MagicMock(name="TracerProvider")
    tracing._provider = fake_provider

    tracing.shutdown_tracing()

    fake_provider.shutdown.assert_called_once_with()
    assert tracing._provider is None

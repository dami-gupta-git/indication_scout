"""Step 2 — auto-instrument ONE real LangChain LLM call and watch its span land in Langfuse.

Run:  python scripts/otel_hello_llm.py
Then open Langfuse → project `indication_scout` → Tracing. You'll see a trace containing an
LLM span with the model name, INPUT/OUTPUT TOKEN COUNTS, and latency — none of which we wrote
by hand. That auto-capture is the whole point of OpenInference.

What's new vs. scripts/otel_hello.py (Step 1):
  - one extra dependency: openinference-instrumentation-langchain
  - one extra line:        LangChainInstrumentor().instrument(tracer_provider=provider)
  - one real LLM call:     ChatAnthropic(...).invoke(...)
Everything else (Resource → TracerProvider → BatchSpanProcessor → OTLPSpanExporter → Langfuse)
is identical to Step 1.

Makes one cheap Anthropic call (~fractions of a cent). Reads all secrets from .env:
  LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL, ANTHROPIC_API_KEY
"""

import base64
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("otel_hello_llm")
# Quiet the per-request HTTP chatter so our own lines are readable.
for noisy in ("httpx", "httpcore", "urllib3", "anthropic", "opentelemetry"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def main() -> None:
    public_key = os.environ["LANGFUSE_PUBLIC_KEY"]
    secret_key = os.environ["LANGFUSE_SECRET_KEY"]
    base_url = os.environ["LANGFUSE_BASE_URL"].rstrip("/")

    auth_token = base64.b64encode(
        f"{public_key}:{secret_key}".encode("utf-8")
    ).decode("ascii")
    endpoint = f"{base_url}/api/public/otel/v1/traces"

    # --- identical OTel bootstrap to Step 1 ---------------------------------------------
    resource = Resource.create({"service.name": "indication-scout-hello-llm"})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=endpoint,
                headers={"Authorization": f"Basic {auth_token}"},
            )
        )
    )
    trace.set_tracer_provider(provider)

    # --- THE NEW LINE: hook LangChain so every LLM call auto-emits a span ----------------
    # This patches LangChain's callback system. From here on, any ChatAnthropic/.invoke()
    # produces an LLM span (model, prompt/response, token usage, latency) with NO span code.
    LangChainInstrumentor().instrument(tracer_provider=provider)

    # --- one real LLM call ---------------------------------------------------------------
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",  # cheapest model — this is just a smoke test
        temperature=0,
        max_tokens=64,
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    logger.info("Calling the LLM...")
    response = llm.invoke("Say hello in exactly five words.")
    logger.info("LLM said: %s", response.content)
    # The token usage that will appear on the span is also available locally here:
    logger.info("usage_metadata: %s", response.usage_metadata)

    provider.shutdown()  # flush before exit
    logger.info(
        "Done. Langfuse → project indication_scout → Tracing: look for an LLM span with "
        "token counts under service 'indication-scout-hello-llm'."
    )


if __name__ == "__main__":
    main()

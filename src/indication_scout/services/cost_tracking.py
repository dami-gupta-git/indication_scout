"""Per-attempt LLM token and cost accounting."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from decimal import Decimal
from threading import Lock
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)

_MILLION = Decimal("1000000")


@dataclass(frozen=True)
class ModelPrices:
    """USD prices per million tokens for one Anthropic model."""

    input: Decimal
    output: Decimal
    cache_write_5m: Decimal
    cache_write_1h: Decimal
    cache_read: Decimal


# Anthropic Claude API list prices verified on 2026-09-12. Unknown models are
# recorded by token count but do not receive an inferred monetary cost.
MODEL_PRICES: dict[str, ModelPrices] = {
    "claude-sonnet-4-6": ModelPrices(
        input=Decimal("3"),
        output=Decimal("15"),
        cache_write_5m=Decimal("3.75"),
        cache_write_1h=Decimal("6"),
        cache_read=Decimal("0.30"),
    ),
    "claude-opus-4-6": ModelPrices(
        input=Decimal("5"),
        output=Decimal("25"),
        cache_write_5m=Decimal("6.25"),
        cache_write_1h=Decimal("10"),
        cache_read=Decimal("0.50"),
    ),
}


@dataclass
class UsageTotals:
    """Token and known-cost totals for a run or candidate."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_5m_tokens: int = 0
    cache_write_1h_tokens: int = 0
    cost_usd: Decimal = Decimal("0")
    unpriced_models: set[str] = field(default_factory=set)

    def add(self, usage: LlmUsage) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_read_tokens += usage.cache_read_tokens
        self.cache_write_5m_tokens += usage.cache_write_5m_tokens
        self.cache_write_1h_tokens += usage.cache_write_1h_tokens
        if usage.cost_usd is None:
            self.unpriced_models.add(usage.model)
        else:
            self.cost_usd += usage.cost_usd


@dataclass(frozen=True)
class LlmUsage:
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_5m_tokens: int
    cache_write_1h_tokens: int
    cost_usd: Decimal | None


@dataclass(frozen=True)
class CostSnapshot:
    total: UsageTotals
    overhead: UsageTotals
    candidates: dict[str, UsageTotals]


class CostTracker:
    """Accumulate usage under the current asynchronous candidate scope."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._total = UsageTotals()
        self._overhead = UsageTotals()
        self._candidates: dict[str, UsageTotals] = defaultdict(UsageTotals)

    def record(self, usage: LlmUsage, candidate: str | None) -> None:
        with self._lock:
            self._total.add(usage)
            if candidate is None:
                self._overhead.add(usage)
            else:
                self._candidates[candidate].add(usage)

    def snapshot(self) -> CostSnapshot:
        with self._lock:
            return CostSnapshot(
                total=_copy_totals(self._total),
                overhead=_copy_totals(self._overhead),
                candidates={
                    name: _copy_totals(totals)
                    for name, totals in self._candidates.items()
                },
            )


_tracker: ContextVar[CostTracker | None] = ContextVar("cost_tracker", default=None)
_candidate: ContextVar[str | None] = ContextVar("cost_candidate", default=None)


def bind_cost_tracker(tracker: CostTracker) -> Token[CostTracker | None]:
    return _tracker.set(tracker)


def reset_cost_tracker(token: Token[CostTracker | None]) -> None:
    _tracker.reset(token)


@contextmanager
def candidate_cost_scope(candidate: str) -> Iterator[None]:
    token = _candidate.set(candidate.strip())
    try:
        yield
    finally:
        _candidate.reset(token)


def calculate_usage(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_5m_tokens: int,
    cache_write_1h_tokens: int,
) -> LlmUsage:
    prices = MODEL_PRICES.get(model)
    cost: Decimal | None = None
    if prices is not None:
        cost = (
            Decimal(input_tokens) * prices.input
            + Decimal(output_tokens) * prices.output
            + Decimal(cache_read_tokens) * prices.cache_read
            + Decimal(cache_write_5m_tokens) * prices.cache_write_5m
            + Decimal(cache_write_1h_tokens) * prices.cache_write_1h
        ) / _MILLION
    return LlmUsage(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_5m_tokens=cache_write_5m_tokens,
        cache_write_1h_tokens=cache_write_1h_tokens,
        cost_usd=cost,
    )


def record_usage(usage: LlmUsage) -> None:
    tracker = _tracker.get()
    if tracker is not None:
        tracker.record(usage, _candidate.get())


def record_anthropic_response(response: Any) -> None:
    if _tracker.get() is None:
        return
    try:
        usage = response.usage
        cache_creation = usage.cache_creation
        cache_write_5m = (
            cache_creation.ephemeral_5m_input_tokens if cache_creation else 0
        )
        cache_write_1h = (
            cache_creation.ephemeral_1h_input_tokens if cache_creation else 0
        )
        if cache_write_5m == 0 and cache_write_1h == 0:
            cache_write_5m = usage.cache_creation_input_tokens or 0
        record_usage(
            calculate_usage(
                model=response.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_input_tokens or 0,
                cache_write_5m_tokens=cache_write_5m,
                cache_write_1h_tokens=cache_write_1h,
            )
        )
    except Exception:  # noqa: BLE001 - accounting must not break an analysis
        logger.exception("Failed to record direct Anthropic usage")


class CostTrackingCallback(BaseCallbackHandler):
    """Capture usage metadata from LangChain model responses."""

    def __init__(self, requested_model: str) -> None:
        self._requested_model = requested_model

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generation_list in response.generations:
            for generation in generation_list:
                message = getattr(generation, "message", None)
                metadata = getattr(message, "usage_metadata", None) or {}
                if not metadata:
                    continue
                response_metadata = getattr(message, "response_metadata", None) or {}
                model = response_metadata.get("model_name", self._requested_model)
                details = metadata.get("input_token_details") or {}
                cache_write_5m = int(details.get("ephemeral_5m_input_tokens", 0))
                cache_write_1h = int(details.get("ephemeral_1h_input_tokens", 0))
                if cache_write_5m == 0 and cache_write_1h == 0:
                    cache_write_5m = int(details.get("cache_creation", 0))
                record_usage(
                    calculate_usage(
                        model=model,
                        input_tokens=int(metadata.get("input_tokens", 0)),
                        output_tokens=int(metadata.get("output_tokens", 0)),
                        cache_read_tokens=int(details.get("cache_read", 0)),
                        cache_write_5m_tokens=cache_write_5m,
                        cache_write_1h_tokens=cache_write_1h,
                    )
                )


def _copy_totals(totals: UsageTotals) -> UsageTotals:
    return UsageTotals(
        input_tokens=totals.input_tokens,
        output_tokens=totals.output_tokens,
        cache_read_tokens=totals.cache_read_tokens,
        cache_write_5m_tokens=totals.cache_write_5m_tokens,
        cache_write_1h_tokens=totals.cache_write_1h_tokens,
        cost_usd=totals.cost_usd,
        unpriced_models=set(totals.unpriced_models),
    )

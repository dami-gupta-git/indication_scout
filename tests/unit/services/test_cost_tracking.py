from decimal import Decimal
from types import SimpleNamespace

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from indication_scout.services.cost_tracking import (
    CostTracker,
    CostTrackingCallback,
    bind_cost_tracker,
    calculate_usage,
    candidate_cost_scope,
    record_anthropic_response,
    record_usage,
    reset_cost_tracker,
)


def test_calculate_usage_prices_all_token_categories():
    usage = calculate_usage(
        model="claude-sonnet-4-6",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_write_5m_tokens=1_000_000,
        cache_write_1h_tokens=1_000_000,
    )

    assert usage.model == "claude-sonnet-4-6"
    assert usage.input_tokens == 1_000_000
    assert usage.output_tokens == 1_000_000
    assert usage.cache_read_tokens == 1_000_000
    assert usage.cache_write_5m_tokens == 1_000_000
    assert usage.cache_write_1h_tokens == 1_000_000
    assert str(usage.cost_usd) == "28.05"


def test_tracker_separates_overhead_and_candidate_usage():
    tracker = CostTracker()
    token = bind_cost_tracker(tracker)
    try:
        overhead = calculate_usage(
            model="claude-sonnet-4-6",
            input_tokens=100,
            output_tokens=10,
            cache_read_tokens=0,
            cache_write_5m_tokens=0,
            cache_write_1h_tokens=0,
        )
        candidate = calculate_usage(
            model="claude-opus-4-6",
            input_tokens=200,
            output_tokens=20,
            cache_read_tokens=50,
            cache_write_5m_tokens=10,
            cache_write_1h_tokens=0,
        )
        record_usage(overhead)
        with candidate_cost_scope("heart failure"):
            record_usage(candidate)
    finally:
        reset_cost_tracker(token)

    snapshot = tracker.snapshot()
    assert snapshot.total.input_tokens == 300
    assert snapshot.total.output_tokens == 30
    assert snapshot.overhead.input_tokens == 100
    assert snapshot.overhead.output_tokens == 10
    assert list(snapshot.candidates) == ["heart failure"]
    assert snapshot.candidates["heart failure"].input_tokens == 200
    assert snapshot.candidates["heart failure"].output_tokens == 20
    assert snapshot.candidates["heart failure"].cache_read_tokens == 50
    assert snapshot.candidates["heart failure"].cache_write_5m_tokens == 10
    assert snapshot.candidates["heart failure"].cache_write_1h_tokens == 0
    assert not snapshot.total.unpriced_models


def test_direct_anthropic_response_records_cache_usage():
    tracker = CostTracker()
    token = bind_cost_tracker(tracker)
    response = SimpleNamespace(
        model="claude-sonnet-4-6",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=20,
            cache_read_input_tokens=30,
            cache_creation_input_tokens=50,
            cache_creation=SimpleNamespace(
                ephemeral_5m_input_tokens=40,
                ephemeral_1h_input_tokens=10,
            ),
        ),
    )
    try:
        record_anthropic_response(response)
    finally:
        reset_cost_tracker(token)

    total = tracker.snapshot().total
    assert total.input_tokens == 10
    assert total.output_tokens == 20
    assert total.cache_read_tokens == 30
    assert total.cache_write_5m_tokens == 40
    assert total.cache_write_1h_tokens == 10
    assert total.cost_usd == Decimal("0.000549")
    assert not total.unpriced_models


def test_langchain_callback_records_response_usage():
    tracker = CostTracker()
    token = bind_cost_tracker(tracker)
    response = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="done",
                        response_metadata={"model_name": "claude-sonnet-4-6"},
                        usage_metadata={
                            "input_tokens": 100,
                            "output_tokens": 25,
                            "total_tokens": 125,
                            "input_token_details": {
                                "cache_read": 20,
                                "ephemeral_5m_input_tokens": 10,
                            },
                        },
                    )
                )
            ]
        ]
    )
    try:
        CostTrackingCallback("claude-opus-4-6").on_llm_end(response)
    finally:
        reset_cost_tracker(token)

    total = tracker.snapshot().total
    assert total.input_tokens == 100
    assert total.output_tokens == 25
    assert total.cache_read_tokens == 20
    assert total.cache_write_5m_tokens == 10
    assert total.cache_write_1h_tokens == 0
    assert total.cost_usd == Decimal("0.0007185")
    assert not total.unpriced_models

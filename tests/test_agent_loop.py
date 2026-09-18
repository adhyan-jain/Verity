"""
Unit tests for Hand-Rolled Investigation Loop (agent/loop.py).
Verifies end-to-end investigation, callback streaming, and Case schema integrity.
"""

import pytest
from agent.loop import run_investigation_loop


def test_investigation_loop_card_fraud():
    case = run_investigation_loop(
        case_id="CASE-CARD-001",
        primary_transaction_id="TX-CARD-9842",
        tier_origin="real_card"
    )

    assert case["case_id"] == "CASE-CARD-001"
    assert case["tier_origin"] == "real_card"
    assert case["primary_transaction_id"] == "TX-CARD-9842"
    assert 0.0 <= case["risk_score"] <= 1.0
    assert len(case["trace_events"]) == 2

    # Event 1 must be get_transaction
    assert case["trace_events"][0]["tool_called"] == "get_transaction"
    # Event 2 must be get_shap_explanation
    assert case["trace_events"][1]["tool_called"] == "get_shap_explanation"

    # Narrative must be strictly non-empty and assembled from trace sentences
    assert len(case["narrative"]) > 20
    assert case["trace_events"][0]["narration_sentence"] in case["narrative"]
    assert case["trace_events"][1]["narration_sentence"] in case["narrative"]


def test_investigation_loop_ledger_anomaly():
    case = run_investigation_loop(
        case_id="CASE-LEDGER-002",
        primary_transaction_id="TX-LEDGER-3011",
        tier_origin="real_ledger"
    )

    assert case["case_id"] == "CASE-LEDGER-002"
    assert case["tier_origin"] == "real_ledger"
    assert len(case["trace_events"]) == 2
    assert case["trace_events"][0]["tool_called"] == "get_transaction"
    assert case["trace_events"][1]["tool_called"] == "walk_graph"
    assert "ACC-1092" in case["trace_events"][1]["narration_sentence"]


def test_investigation_loop_synthetic_network():
    case = run_investigation_loop(
        case_id="CASE-SYNTH-003",
        primary_transaction_id="TX-SYNTH-5501",
        tier_origin="synthetic_network"
    )

    assert case["case_id"] == "CASE-SYNTH-003"
    assert case["tier_origin"] == "synthetic_network"
    assert len(case["trace_events"]) == 2
    assert case["trace_events"][1]["tool_called"] == "walk_graph"
    assert "round-tripping" in case["narrative"].lower() or "circular" in case["narrative"].lower()


def test_investigation_loop_streaming_callback():
    streamed_events = []

    def on_step(event):
        streamed_events.append(event)

    case = run_investigation_loop(
        case_id="CASE-STREAM-004",
        primary_transaction_id="TX-CARD-9842",
        tier_origin="real_card",
        on_step_callback=on_step
    )

    assert len(streamed_events) == 2
    assert streamed_events[0]["tool_called"] == "get_transaction"
    assert streamed_events[1]["tool_called"] == "get_shap_explanation"

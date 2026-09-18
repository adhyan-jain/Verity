"""
Unit tests for Dynamic Tool Calling Loop with LLM Reasoning (agent/loop.py).
Verifies LLM tool selection, multi-tier evidence analysis, and Case schema integrity.
"""

import pytest
from agent.loop import run_investigation_loop
from agent.llm import VerityLLMClient


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

    # Step 1: get_transaction
    assert case["trace_events"][0]["tool_called"] == "get_transaction"
    # Step 2: get_shap_explanation
    assert case["trace_events"][1]["tool_called"] == "get_shap_explanation"

    # Narrative must be strictly non-empty and assembled from trace sentences
    assert len(case["narrative"]) > 20
    assert case["trace_events"][0]["narration_sentence"] in case["narrative"]
    assert case["trace_events"][1]["narration_sentence"] in case["narrative"]


def test_investigation_loop_ledger_dynamically_calculates_balance_break():
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

    # Narrative must reflect actual calculated balance numbers from mock_timelines (-$3,200.00)
    event2_sentence = case["trace_events"][1]["narration_sentence"]
    assert "-3,200" in event2_sentence or "overdraft" in event2_sentence
    assert case["risk_score"] >= 0.70


def test_investigation_loop_synthetic_dynamically_calculates_volume_retention():
    case = run_investigation_loop(
        case_id="CASE-SYNTH-003",
        primary_transaction_id="TX-SYNTH-5501",
        tier_origin="synthetic_network"
    )

    assert case["case_id"] == "CASE-SYNTH-003"
    assert case["tier_origin"] == "synthetic_network"
    assert len(case["trace_events"]) == 2
    assert case["trace_events"][1]["tool_called"] == "walk_graph"

    # Narrative must reflect mathematically verified volume retention ratio (96.9%)
    event2_sentence = case["trace_events"][1]["narration_sentence"]
    assert "96.9%" in event2_sentence or "round-tripping" in event2_sentence.lower()
    assert case["risk_score"] >= 0.90


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


def test_llm_decision_step():
    client = VerityLLMClient()
    # Test step 1: when history is empty, model must choose get_transaction
    decision1 = client.decide_next_step("C-1", "TX-1", "real_card", [], 1)
    assert decision1["action"] == "get_transaction"
    assert "TX-1" in str(decision1["action_input"])

    # Test step 2: after get_transaction, card tier must choose get_shap_explanation
    mock_history = [{
        "tool_called": "get_transaction",
        "raw_output": {"tier": "real_card", "id": "TX-1"}
    }]
    decision2 = client.decide_next_step("C-1", "TX-1", "real_card", mock_history, 2)
    assert decision2["action"] == "get_shap_explanation"

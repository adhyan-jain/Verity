"""
Unit tests for OpenRouter & Provider-Agnostic LLM Configuration (agent/llm.py).
Verifies OpenRouter configuration resolution, response parsing across formats,
safe fallback when keys are omitted, strict credential masking (never logged or leaked),
and execution of all 4 investigative tools including counterfactual.
"""

import os
import logging
import pytest
from unittest.mock import patch, MagicMock

from agent.llm import (
    VerityLLMClient,
    resolve_llm_config,
    mask_key,
    parse_llm_response,
    ALLOWED_TOOLS
)
from agent.loop import run_investigation_loop


def test_openrouter_configuration_loading(monkeypatch):
    """Verifies that OpenRouter provider, key, base URL, and model load correctly from env."""
    monkeypatch.setenv("VERITY_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("VERITY_LLM_API_KEY", "sk-or-v1-testkey1234567890abcdef")
    monkeypatch.setenv("VERITY_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("VERITY_LLM_MODEL", "openrouter/free")
    monkeypatch.setenv("VERITY_LLM_TIMEOUT", "12.0")

    client = VerityLLMClient()
    assert client.provider == "openrouter"
    assert client.api_key == "sk-or-v1-testkey1234567890abcdef"
    assert client.base_url == "https://openrouter.ai/api/v1"
    assert client.model == "openrouter/free"
    assert client.timeout == 12.0


def test_openrouter_default_base_url_and_model(monkeypatch):
    """Verifies that setting provider=openrouter automatically defaults to OpenRouter endpoints."""
    monkeypatch.setenv("VERITY_LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("VERITY_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("VERITY_LLM_MODEL", raising=False)

    provider, key, base_url, model, timeout = resolve_llm_config()
    assert provider == "openrouter"
    assert base_url == "https://openrouter.ai/api/v1"
    assert model == "openrouter/free"


def test_auto_detect_openrouter_from_key_or_url(monkeypatch):
    """Auto-detects openrouter provider when sk-or- key prefix or openrouter.ai URL is supplied."""
    monkeypatch.delenv("VERITY_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("VERITY_LLM_API_KEY", "sk-or-v1-abc123456")
    monkeypatch.delenv("VERITY_LLM_BASE_URL", raising=False)

    provider, key, base_url, model, _ = resolve_llm_config()
    assert provider == "openrouter"
    assert base_url == "https://openrouter.ai/api/v1"


def test_missing_key_falls_back_safely(monkeypatch):
    """Verifies that when no API key is set, client falls back to built-in reasoning without network calls."""
    monkeypatch.setenv("VERITY_LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("VERITY_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = VerityLLMClient()
    assert client.api_key == ""

    # Step 1 decision must fall back to built-in deterministic reasoning
    decision = client.decide_next_step(
        case_id="CASE-TEST-001",
        primary_tx_id="TX-CARD-9842",
        tier_origin="real_card",
        history=[],
        step_number=1
    )
    assert decision["action"] == "get_transaction"
    assert decision["action_input"] == {"transaction_id": "TX-CARD-9842"}
    assert "thought" in decision
    assert len(decision["thought"]) > 0

    # Chat answer must fall back to contextual reasoning without network call
    chat_answer = client.generate_chat_answer(
        query="What happened?",
        case_id="CASE-TEST-001",
        trace_events=[]
    )
    assert "Analyst query received" in chat_answer


def test_api_key_never_returned_in_summary_or_logs(monkeypatch, caplog):
    """Verifies that the API key is never exposed in get_config_summary() or logger output."""
    secret = "sk-or-v1-supersecretkey999888777"
    monkeypatch.setenv("VERITY_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("VERITY_LLM_API_KEY", secret)

    client = VerityLLMClient()
    summary = client.get_config_summary()

    # Must be masked in summary
    assert secret not in str(summary)
    assert summary["api_key_masked"] == "sk-...8777"
    assert summary["api_key_configured"] is True

    # Simulate network failure and verify logs do not leak the key
    with caplog.at_level(logging.WARNING):
        with patch("agent.llm.requests.post", side_effect=Exception("Connection refused")):
            decision = client.decide_next_step(
                case_id="CASE-SEC-001",
                primary_tx_id="TX-CARD-9842",
                tier_origin="real_card",
                history=[],
                step_number=1
            )
            # Must safely fall back to built-in reasoning
            assert decision["action"] == "get_transaction"

        # Check all logged messages
        for record in caplog.records:
            assert secret not in record.getMessage()


def test_response_parser_extracts_all_fields_and_formats():
    """Verifies response parser extracts thought, action, action_input, and candidate_narrative across formats."""
    # 1. Clean JSON
    clean_json = (
        '{"thought": "Need transaction baseline", "action": "get_transaction", '
        '"action_input": {"transaction_id": "TX-CARD-9842"}, "candidate_narrative": null}'
    )
    parsed1 = parse_llm_response(clean_json)
    assert parsed1 is not None
    assert parsed1["thought"] == "Need transaction baseline"
    assert parsed1["action"] == "get_transaction"
    assert parsed1["action_input"] == {"transaction_id": "TX-CARD-9842"}
    assert parsed1["candidate_narrative"] is None

    # 2. Markdown fenced JSON (```json ... ```)
    fenced_json = (
        "Here is the next step:\n```json\n"
        '{\n  "thought": "Evaluate counterfactual with reduced amount",\n'
        '  "action": "counterfactual",\n'
        '  "action_input": {"transaction_id": "TX-CARD-9842", "parameter_overrides": {"Amount": 50.0}},\n'
        '  "candidate_narrative": null\n}\n```'
    )
    parsed2 = parse_llm_response(fenced_json)
    assert parsed2 is not None
    assert parsed2["action"] == "counterfactual"
    assert parsed2["action_input"]["parameter_overrides"]["Amount"] == 50.0

    # 3. Finish step with candidate narrative
    finish_json = (
        '{"thought": "Investigation complete", "action": "finish", '
        '"action_input": {}, "candidate_narrative": "Verified card fraud pattern on TX-CARD-9842."}'
    )
    parsed3 = parse_llm_response(finish_json)
    assert parsed3 is not None
    assert parsed3["action"] == "finish"
    assert parsed3["candidate_narrative"] == "Verified card fraud pattern on TX-CARD-9842."

    # 4. Stringified action_input auto-parsed to dict
    string_input_json = (
        '{"thought": "Checking graph", "action": "walk_graph", '
        '"action_input": "{\\"account_id\\": \\"ACC-1092\\", \\"tier\\": \\"real_ledger\\"}"}'
    )
    parsed4 = parse_llm_response(string_input_json)
    assert parsed4 is not None
    assert parsed4["action"] == "walk_graph"
    assert parsed4["action_input"]["account_id"] == "ACC-1092"

    # 5. Unallowed action rejected safely
    bad_action_json = '{"thought": "Unauthorized tool", "action": "delete_database", "action_input": {}}'
    parsed_bad = parse_llm_response(bad_action_json)
    assert parsed_bad is None


def test_agent_loop_can_select_and_execute_counterfactual():
    """Verifies that the agent loop can dispatch and execute the counterfactual tool cleanly."""
    mock_client = MagicMock(spec=VerityLLMClient)

    # Step 1: get_transaction
    # Step 2: counterfactual
    # Step 3: finish
    mock_client.decide_next_step.side_effect = [
        {
            "thought": "Fetch primary transaction details",
            "action": "get_transaction",
            "action_input": {"transaction_id": "TX-CARD-9842"},
            "candidate_narrative": None
        },
        {
            "thought": "Evaluate counterfactual scenario reducing amount to $50",
            "action": "counterfactual",
            "action_input": {
                "transaction_id": "TX-CARD-9842",
                "parameter_overrides": {"Amount": 50.0}
            },
            "candidate_narrative": None
        },
        {
            "thought": "Finish investigation with grounded conclusion",
            "action": "finish",
            "action_input": {},
            "candidate_narrative": "Retrieved transaction TX-CARD-9842 for $4,850.00 processed on rail creditcard.csv."
        }
    ]

    case = run_investigation_loop(
        case_id="CASE-CF-001",
        primary_transaction_id="TX-CARD-9842",
        tier_origin="real_card",
        llm_client=mock_client
    )

    assert case["case_id"] == "CASE-CF-001"
    assert len(case["trace_events"]) == 2
    assert case["trace_events"][0]["tool_called"] == "get_transaction"
    assert case["trace_events"][1]["tool_called"] == "counterfactual"
    assert "Amount" in str(case["trace_events"][1]["tool_input"])
    assert "Counterfactual evaluation" in case["trace_events"][1]["tool_output_summary"]
    assert len(case["narrative"]) > 0


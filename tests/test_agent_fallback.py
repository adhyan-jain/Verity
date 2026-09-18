"""
Unit tests for Latency Fallback & Cached Q&A (agent/fallback.py).
Verifies disclosure notices and trigger pattern matching for the 4 core judge questions.
"""

from agent.fallback import (
    STANDARD_FALLBACK_NOTICE,
    execute_with_latency_guard,
    get_cached_answer,
)


def test_fallback_q1_why_flagged():
    ans = get_cached_answer("Can you explain why was this flagged?")
    assert ans is not None
    assert ans["is_fallback"] is True
    assert ans["fallback_notice"] == STANDARD_FALLBACK_NOTICE
    assert ans["question_id"] == "q1_why_flagged"
    assert "TX-CARD-9842" in ans["response"]


def test_fallback_q2_counterfactual():
    ans = get_cached_answer("what if the amount were different?")
    assert ans is not None
    assert ans["is_fallback"] is True
    assert ans["question_id"] == "q2_counterfactual_amount"
    assert "Reducing the transaction amount" in ans["response"]


def test_fallback_q3_similar_cases():
    ans = get_cached_answer("Please show me a similar case in the system")
    assert ans is not None
    assert ans["is_fallback"] is True
    assert ans["question_id"] == "q3_similar_cases"
    assert "CASE-CARD-082" in ans["response"]


def test_fallback_q4_counterparty_not_flagged():
    ans = get_cached_answer(
        "why wasn't this other account flagged for this transaction?"
    )
    assert ans is not None
    assert ans["is_fallback"] is True
    assert ans["question_id"] == "q4_why_other_account_not_flagged"
    assert "ACC-1080" in ans["response"]


def test_fallback_unmatched_query():
    ans = get_cached_answer("What is the capital of France?")
    assert ans is None


def test_latency_guard_timeout():
    import time

    def slow_task():
        time.sleep(0.05)
        return {"data": "slow"}

    # timeout_seconds well below the task's real sleep forces a deterministic timeout
    res = execute_with_latency_guard(
        slow_task, query="why was this flagged", timeout_seconds=0.01
    )
    assert res.get("is_fallback") is True
    assert "Execution exceeded latency limit" in res.get("fallback_reason", "")

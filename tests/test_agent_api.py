"""
Integration tests for Agent API Service (agent/api.py).
Tests REST endpoints, SSE trace streaming, model-backed counterfactuals,
and conversational AI chat with latency guard watchdog.
"""

import os

from fastapi.testclient import TestClient

os.environ.setdefault("AGENT_API_KEY", "test-key")

from agent.api import app

API_KEY_HEADERS = {"X-API-Key": os.environ["AGENT_API_KEY"]}
client = TestClient(app, headers=API_KEY_HEADERS)


def test_healthcheck():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "verity-agent-core"
    assert data["version"] == "2.0.0"


def test_api_investigate_card_case():
    payload = {
        "case_id": "CASE-CARD-001",
        "transaction_id": "TX-CARD-9842",
        "tier_origin": "real_card",
    }
    response = client.post("/api/v1/agent/investigate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["case_id"] == "CASE-CARD-001"
    assert data["tier_origin"] == "real_card"
    assert data["status"] == "investigating"
    assert len(data["trace_events"]) == 2
    assert "narrative" in data
    assert len(data["narrative"]) > 0


def test_api_trace_stream_json():
    # Standard JSON poll
    response = client.get("/api/v1/agent/trace-stream/CASE-CARD-001")
    assert response.status_code == 200
    events = response.json()
    assert isinstance(events, list)
    assert len(events) >= 2
    assert events[0]["tool_called"] == "get_transaction"


def test_api_trace_stream_sse():
    # Real Server-Sent Events (SSE) stream
    headers = {"Accept": "text/event-stream"}
    response = client.get("/api/v1/agent/trace-stream/CASE-CARD-001", headers=headers)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "data:" in response.text
    assert "stream_status" in response.text or "EVT-" in response.text


def test_api_counterfactual_model_backed():
    payload = {
        "transaction_id": "TX-CARD-9842",
        "parameter_overrides": {"Amount": 50.0},
    }
    response = client.post("/api/v1/agent/counterfactual", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "TX-CARD-9842"
    assert data["original_risk_score"] >= 0.80
    assert data["recalculated_risk_score"] < 0.50
    assert data["recalculated_verdict"] == "clear"
    assert "feature_attribution_deltas" in data


def test_api_chat_cached_benchmark_question():
    payload = {"case_id": "CASE-CARD-001", "query": "why was this flagged"}
    response = client.post("/api/v1/agent/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fallback"] is True
    assert "TX-CARD-9842" in data["response"]
    assert "Using a prepared benchmark answer" in data["fallback_notice"]


def test_api_chat_contextual_ai_query():
    # Analyst asks question about active case
    payload = {
        "case_id": "CASE-CARD-001",
        "query": "Summarize the primary risk findings for this card transaction.",
    }
    response = client.post("/api/v1/agent/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fallback"] is False
    assert len(data["response"]) > 20


def test_api_chat_latency_watchdog_trigger():
    # Force latency fallback via simulated delay > 20s using a small
    # simulated latency with a fast timeout in the guard itself.
    import time

    from agent.fallback import execute_with_latency_guard

    res = execute_with_latency_guard(
        task_func=lambda: time.sleep(0.08) or {"slow": True},
        query="why was this flagged",
        timeout_seconds=0.01,
    )
    assert res.get("is_fallback") is True
    assert "Execution exceeded latency limit" in res.get("fallback_reason", "")


def test_api_get_queue_paginated():
    # Test unpaginated/default behavior
    resp_all = client.post("/api/v1/aml/queue", json={})
    assert resp_all.status_code == 200
    data_all = resp_all.json()
    total = data_all.get("total_records", len(data_all.get("records", [])))

    # Test paginated behavior
    resp_p1 = client.post("/api/v1/aml/queue", json={"page": 1, "page_size": 2})
    assert resp_p1.status_code == 200
    data_p1 = resp_p1.json()
    assert data_p1["current_page"] == 1
    assert data_p1["page_size"] == 2
    assert len(data_p1["records"]) <= 2
    if total > 0:
        assert data_p1["total_pages"] >= 1


def test_api_get_timeline_paginated():
    account_id = "409000493210"
    resp_p1 = client.get(f"/api/v1/aml/timeline/{account_id}?page=1&page_size=5")
    assert resp_p1.status_code == 200
    data_p1 = resp_p1.json()
    assert data_p1["current_page"] == 1
    assert data_p1["page_size"] == 5
    assert len(data_p1["timeline"]) <= 5
    if data_p1["total_txns"] > 5:
        assert data_p1["total_pages"] > 1
        resp_p2 = client.get(f"/api/v1/aml/timeline/{account_id}?page=2&page_size=5")
        assert resp_p2.status_code == 200
        data_p2 = resp_p2.json()
        assert data_p2["current_page"] == 2
        # Ensure page 1 and page 2 don't overlap IDs
        p1_ids = {tx["id"] for tx in data_p1["timeline"]}
        p2_ids = {tx["id"] for tx in data_p2["timeline"]}
        assert p1_ids.isdisjoint(p2_ids)


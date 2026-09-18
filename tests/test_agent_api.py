"""
Integration tests for Agent API Service (agent/api.py).
Tests REST endpoints powering Person D's dashboard cockpit.
"""

import pytest
from fastapi.testclient import TestClient
from agent.api import app

client = TestClient(app)


def test_healthcheck():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "verity-agent-core"


def test_api_investigate_card_case():
    payload = {
        "case_id": "CASE-CARD-001",
        "transaction_id": "TX-CARD-9842",
        "tier_origin": "real_card"
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


def test_api_trace_stream():
    # Retrieve trace stream for the investigated case
    response = client.get("/api/v1/agent/trace-stream/CASE-CARD-001")
    assert response.status_code == 200
    events = response.json()
    assert isinstance(events, list)
    assert len(events) >= 2
    assert events[0]["tool_called"] == "get_transaction"


def test_api_counterfactual():
    payload = {
        "transaction_id": "TX-CARD-9842",
        "parameter_overrides": {
            "Amount": 120.0
        }
    }
    response = client.post("/api/v1/agent/counterfactual", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "TX-CARD-9842"
    assert data["original_risk_score"] > 0.8
    assert data["recalculated_risk_score"] < 0.5
    assert data["recalculated_verdict"] == "clear"


def test_api_chat_cached_question():
    payload = {
        "case_id": "CASE-CARD-001",
        "query": "why was this flagged"
    }
    response = client.post("/api/v1/agent/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fallback"] is True
    assert "TX-CARD-9842" in data["response"]
    assert "Using a prepared benchmark answer" in data["fallback_notice"]


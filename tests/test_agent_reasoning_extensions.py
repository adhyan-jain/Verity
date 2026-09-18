"""
Automated Unit & Integration Test Suite for Person C Reasoning Extensions:
- Prosecutor / Defender Adjudication (agent/prosecutor_defender.py)
- Live Trust Score Tracking (agent/trust_score.py)
- Screen 3 Breakdown Panel with Gate 1 Compliance (agent/breakdown.py)
- Customer-Scoped Chat, Counterfactual & Forward Simulation (agent/simulation.py)
- New AML Decision Support API Endpoints in agent/api.py
"""

import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient

os.environ.setdefault("AGENT_API_KEY", "test-key")
API_KEY_HEADERS = {"X-API-Key": os.environ["AGENT_API_KEY"]}

from agent.api import app
from agent.prosecutor_defender import adjudicate_case, _resolution_to_dict
from agent.trust_score import TrustScoreSession, tag_claim, get_session_score
from agent.breakdown import generate_account_breakdown
from agent.simulation import (
    run_customer_counterfactual,
    run_forward_simulation,
    handle_scoped_customer_chat,
    wrap_score,
)


@pytest.fixture(scope="module")
def client():
    return TestClient(app, headers=API_KEY_HEADERS)


@pytest.fixture
def mock_bank_ledger():
    now = pd.Timestamp("2018-06-01")
    rows = [
        {
            "id": "TX-MOCK-01",
            "account_id": "ACC-TEST-100",
            "datetime": now - pd.Timedelta(days=60),
            "timestamp": (now - pd.Timedelta(days=60)).strftime("%Y-%m-%d"),
            "amount": 500.0,
            "direction": "debit",
            "payment_rail": "NEFT",
            "raw_narration": "MONTHLY OFFICE RENT",
            "balance": 5000.0,
        },
        {
            "id": "TX-MOCK-02",
            "account_id": "ACC-TEST-100",
            "datetime": now - pd.Timedelta(days=30),
            "timestamp": (now - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
            "amount": 500.0,
            "direction": "debit",
            "payment_rail": "NEFT",
            "raw_narration": "MONTHLY OFFICE RENT",
            "balance": 4500.0,
        },
        {
            "id": "TX-MOCK-03",
            "account_id": "ACC-TEST-100",
            "datetime": now,
            "timestamp": now.strftime("%Y-%m-%d"),
            "amount": 500.0,
            "direction": "debit",
            "payment_rail": "NEFT",
            "raw_narration": "MONTHLY OFFICE RENT",
            "balance": 4000.0,
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Prosecutor / Defender Adjudication
# ---------------------------------------------------------------------------
def test_prosecutor_defender_recurring_downgrade(mock_bank_ledger):
    curr = mock_bank_ledger.iloc[-1]
    res = adjudicate_case(
        tx_row=curr,
        acct_df=mock_bank_ledger,
        risk_score=0.72,
        all_scores=[0.2] * 50 + [0.72],
        structuring_flags=[],
        log=False,
    )
    d = _resolution_to_dict(res)
    assert d["transaction_id"] == "TX-MOCK-03"
    assert "prosecutor" in d
    assert "defender" in d
    assert len(d["prosecutor"]["arguments"]) >= 1
    # Defender should match recurring payment (same amount & narration)
    assert len(d["defender"]["grounded_reasons"]) >= 1
    assert d["verdict"] in ("downgraded", "cleared")


# ---------------------------------------------------------------------------
# 2. Live Trust Score Tracking
# ---------------------------------------------------------------------------
def test_live_trust_score_grounding_filter():
    session = TrustScoreSession("CASE-TEST-TRUST-01")
    trace = [
        {
            "event_id": "EVT-1",
            "narration_sentence": "Retrieved ledger entry TX-MOCK-03 for $500.00 processed on NEFT rail.",
            "tool_output_summary": "id TX-MOCK-03 amount 500.00 rail NEFT",
        }
    ]
    
    # Grounded claim matching trace
    tag1 = session.tag_claim(
        "Retrieved ledger entry TX-MOCK-03 for $500.00 processed on NEFT rail.",
        trace,
    )
    assert tag1.grounded is True

    # Hallucinated / unsupported entity claim
    tag2 = session.tag_claim(
        "Detected unverified offshore shell company in Cayman Islands.",
        trace,
    )
    assert tag2.grounded is False

    state = session.state.to_dict()
    assert state["total_claims"] == 2
    assert state["grounded_claims"] == 1
    assert state["grounded_pct"] == 50.0
    assert "50% of this session's claims are grounded" in state["summary"]


# ---------------------------------------------------------------------------
# 3. Screen 3 Breakdown Panel & Gate 1 Compliance
# ---------------------------------------------------------------------------
def test_breakdown_gate1_compliance(mock_bank_ledger):
    breakdown = generate_account_breakdown("ACC-TEST-100", bank_df=mock_bank_ledger)
    assert breakdown["account_id"] == "ACC-TEST-100"
    assert "claims" in breakdown
    assert len(breakdown["claims"]) > 0

    # Gate 1 verification: strict date-only precision, zero time-of-day phrases
    forbidden_phrases = ["3am", "3:00", "after-hours", "overnight", "peak hours", "03:00", "pm", "am"]
    for claim in breakdown["claims"]:
        assert claim["gate1_compliant"] is True
        assert "evidence_row_id" in claim
        assert "source_tag" in claim
        sentence_lower = claim["sentence"].lower()
        for phrase in forbidden_phrases:
            assert f" {phrase} " not in f" {sentence_lower} ", f"Gate 1 violation found: '{phrase}' in '{claim['sentence']}'"


# ---------------------------------------------------------------------------
# 4. Scoped Simulation & Counterfactual Recompute
# ---------------------------------------------------------------------------
def test_scoped_simulation_counterfactual(mock_bank_ledger):
    cf = run_customer_counterfactual(
        account_id="ACC-TEST-100",
        transaction_id="TX-MOCK-03",
        new_amount=100.0,
        bank_df=mock_bank_ledger,
    )
    assert cf["transaction_id"] == "TX-MOCK-03"
    assert cf["new_amount"] == 100.0
    assert "recalculated_risk_score" in cf
    assert "conformal_interval" in cf

    # Test forward simulation
    fwd = run_forward_simulation(
        account_id="ACC-TEST-100",
        transaction_id="TX-MOCK-03",
        days_ahead=7,
        bank_df=mock_bank_ledger,
    )
    assert fwd["account_id"] == "ACC-TEST-100"
    assert fwd["days_ahead"] == 7
    assert "forward_risk_score" in fwd

    # Test scoped chat handler
    chat_resp = handle_scoped_customer_chat(
        account_id="ACC-TEST-100",
        query="what if this amount was $100 instead?",
        primary_tx_id="TX-MOCK-03",
        bank_df=mock_bank_ledger,
    )
    assert "ACC-TEST-100" in chat_resp["response"]
    assert chat_resp["account_id"] == "ACC-TEST-100"


# ---------------------------------------------------------------------------
# 5. AML API Endpoints Integration
# ---------------------------------------------------------------------------
def test_aml_api_queue_and_conformal(client):
    resp = client.post("/api/v1/aml/conformal", json={"risk_score": 0.85, "confidence": 0.90})
    assert resp.status_code == 200
    data = resp.json()
    assert "conformal_lo" in data
    assert "conformal_hi" in data
    assert data["conformal_lo"] <= 0.85 <= data["conformal_hi"]

    # Test queue endpoint
    resp_q = client.post("/api/v1/aml/queue", json={"top_n": 5, "with_conformal": True})
    assert resp_q.status_code == 200
    data_q = resp_q.json()
    assert "records" in data_q

    # Test trust score endpoint
    resp_ts = client.post(
        "/api/v1/aml/trust-score/tag",
        json={
            "case_id": "CASE-API-TEST",
            "sentence": "Transaction TX-100 processed on NEFT rail.",
            "trace_events": [{"event_id": "EVT-1", "narration_sentence": "Transaction TX-100 processed on NEFT rail."}],
        },
    )
    assert resp_ts.status_code == 200
    data_ts = resp_ts.json()
    assert data_ts["grounded"] is True

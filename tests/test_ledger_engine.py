"""
Automated Unit & Integration Test Suite for Ledger Engine.
Person B: Tests narration parser, baseline computation, anomaly detection,
visual timeline generation, and Ledger API endpoints.
"""

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("LEDGER_API_KEY", "test-key")

from engines.ledger.anomalies import detect_all_ledger_anomalies
from engines.ledger.api import app
from engines.ledger.parse_narrations import (
    extract_payment_rail,
    get_all_account_summaries,
    parse_bank_ledger,
)
from engines.ledger.reconcile import compute_all_account_baselines
from engines.ledger.timeline import build_account_timeline

API_KEY_HEADERS = {"X-API-Key": os.environ["LEDGER_API_KEY"]}


@pytest.fixture(scope="module")
def parsed_df():
    return parse_bank_ledger()


@pytest.fixture(scope="module")
def client():
    return TestClient(app, headers=API_KEY_HEADERS)


def test_payment_rail_classifier():
    assert extract_payment_rail("RTGS/INDO1234/CORP PMT") == "RTGS"
    assert extract_payment_rail("NEFT-SBIN123-TRANSFER") == "NEFT"
    assert extract_payment_rail("UPI/CR/987654321/GROCERY") == "UPI"
    assert extract_payment_rail("IMPS-P2A-654321") == "IMPS"
    assert extract_payment_rail("FDRL/INTERNAL FUND TRANSFE") == "INTERNAL_TRANSFER"
    assert extract_payment_rail("INDO GIBL Indiaforensic STL01071") == "INDO_GIBL"
    assert extract_payment_rail("CHQ DEPOSIT CLEARING #9921") == "CHEQUE"
    assert extract_payment_rail(None) == "OTHER"


def test_ledger_dataframe_integrity(parsed_df):
    assert len(parsed_df) == 116201
    assert parsed_df["account_id"].nunique() == 10
    required_cols = [
        "id",
        "tier",
        "account_id",
        "timestamp",
        "amount",
        "direction",
        "balance",
        "raw_narration",
        "payment_rail",
    ]
    for col in required_cols:
        assert col in parsed_df.columns
    assert (parsed_df["amount"] >= 0).all()
    assert set(parsed_df["direction"].unique()) == {"debit", "credit"}
    assert (parsed_df["tier"] == "real_ledger").all()


def test_account_summaries(parsed_df):
    summaries = get_all_account_summaries(parsed_df)
    assert len(summaries) == 10
    for s in summaries:
        assert s["total_transactions"] > 0
        assert s["total_debit_volume"] >= 0
        assert s["total_credit_volume"] >= 0
        assert "account_id" in s


def test_account_baselines(parsed_df):
    baselines = compute_all_account_baselines(parsed_df)
    assert len(baselines) == 10

    # Check baseline structure for account 1196428
    b = baselines["1196428"]
    assert b["velocity"]["mean_daily"] > 0
    assert b["velocity"]["p95_daily"] >= b["velocity"]["mean_daily"]
    assert "lower_bound_3sigma" in b["balance"]
    assert 0 <= b["reversals"]["baseline_reversal_rate"] <= 1.0


def test_anomaly_detection_schemas(parsed_df):
    anomalies = detect_all_ledger_anomalies(parsed_df)
    assert len(anomalies) > 0

    valid_types = {"balance_break", "timing_spike", "reversal_outlier"}
    for anom in anomalies:
        assert anom["anomaly_type"] in valid_types
        assert 0.0 <= anom["severity"] <= 1.0
        assert len(anom["evidence_transaction_ids"]) >= 1
        assert "window_start" in anom
        assert "window_end" in anom


def test_timeline_generation(parsed_df):
    timeline = build_account_timeline("1196428", parsed_df)
    assert timeline["account_id"] == "1196428"
    assert timeline["tier"] == "real_ledger"
    assert len(timeline["density_curve"]) > 0
    assert len(timeline["transactions"]) > 0
    assert len(timeline["anomalies"]) > 0

    # Verify density curve fields
    first_pt = timeline["density_curve"][0]
    assert "date" in first_pt
    assert "tx_count" in first_pt
    assert "closing_balance" in first_pt
    assert "has_anomaly" in first_pt


def test_ledger_api_endpoints(client):
    # 1. Health
    r = client.get("/api/v1/ledger/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

    # 2. List accounts
    r = client.get("/api/v1/ledger/accounts")
    assert r.status_code == 200
    assert len(r.json()) == 10

    # 3. Get anomalies
    r = client.get("/api/v1/ledger/anomalies/1196428")
    assert r.status_code == 200
    assert isinstance(r.json(), list)

    # 4. Get timeline
    r = client.get("/api/v1/ledger/timeline/1196428")
    assert r.status_code == 200
    assert r.json()["account_id"] == "1196428"

    # 5. Graph walk steps
    r = client.get("/api/v1/ledger/walk/1196428?limit=10")
    assert r.status_code == 200
    walk = r.json()
    assert len(walk) == 10
    assert walk[0]["tier"] == "real_ledger"
    assert "step_index" in walk[0]

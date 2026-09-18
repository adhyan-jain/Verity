import json
import os

import pytest
from fastapi.testclient import TestClient
from jsonschema import validate

os.environ.setdefault("FRAUD_API_KEY", "test-key")

from engines.fraud.api import app
from engines.fraud.explain import explain_transaction

API_KEY_HEADERS = {"X-API-Key": os.environ["FRAUD_API_KEY"]}


@pytest.fixture(scope="session")
def client():
    return TestClient(app, headers=API_KEY_HEADERS)


@pytest.fixture(scope="session")
def schemas():
    with open("contracts/schemas.json") as f:
        return json.load(f)["definitions"]


def test_health_check(client):
    response = client.get("/api/v1/fraud/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["engine"] == "fraud"
    assert "metrics" in data


def test_get_transaction_valid(client, schemas):
    response = client.get("/api/v1/fraud/transaction/TX-CARD-541")
    assert response.status_code == 200
    data = response.json()
    validate(instance=data, schema=schemas["TransactionRecord"])
    assert data["id"] == "TX-CARD-541"
    assert data["tier"] == "real_card"
    assert data["source_dataset"] == "creditcard.csv"
    assert isinstance(data["amount"], (int, float))


def test_get_transaction_numeric_id(client, schemas):
    response = client.get("/api/v1/fraud/transaction/0")
    assert response.status_code == 200
    data = response.json()
    validate(instance=data, schema=schemas["TransactionRecord"])
    assert data["id"] == "TX-CARD-0"


def test_get_transaction_invalid_id(client):
    response = client.get("/api/v1/fraud/transaction/TX-CARD-9999999")
    assert response.status_code == 404


def test_get_shap_explanation_flagged(client, schemas):
    response = client.get("/api/v1/fraud/explain/TX-CARD-541")
    assert response.status_code == 200
    data = response.json()
    validate(instance=data, schema=schemas["FraudExplanation"])
    assert data["verdict"] == "flagged"
    assert data["risk_score"] > 0.5
    assert len(data["top_factors"]) > 0

    # Verify interpretable flag segregation
    for factor in data["top_factors"]:
        if factor["feature"] in ["Amount", "Time"]:
            assert factor["interpretable"] is True
        elif factor["feature"].startswith("V"):
            assert factor["interpretable"] is False


def test_get_shap_explanation_clear(client, schemas):
    response = client.get("/api/v1/fraud/explain/TX-CARD-0")
    assert response.status_code == 200
    data = response.json()
    validate(instance=data, schema=schemas["FraudExplanation"])
    assert data["verdict"] == "clear"
    assert data["risk_score"] < 0.5


def test_counterfactual_endpoint(client):
    payload = {
        "transaction_id": "TX-CARD-541",
        "parameter_overrides": {
            "Amount": 10.0,
            "Time": 80000.0,
            "V14": 0.0,
        },
    }
    response = client.post("/api/v1/fraud/counterfactual", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "TX-CARD-541"
    assert "original_risk_score" in data
    assert "recalculated_risk_score" in data
    assert "original_verdict" in data
    assert "recalculated_verdict" in data
    assert data["modifications"]["Amount"] == 10.0


def test_list_transactions(client):
    response = client.get("/api/v1/fraud/transactions?limit=10&flagged_only=true")
    assert response.status_code == 200
    data = response.json()
    assert data["limit"] == 10
    assert len(data["transactions"]) == 10
    assert all(tx["is_ground_truth_fraud"] for tx in data["transactions"])


def test_explain_standalone_features():
    features = {
        "Time": 5000.0,
        "Amount": 1200.0,
        "V1": 0.5,
        "V14": -4.0,
    }
    res = explain_transaction(features=features, transaction_id="TX-CUSTOM-1")
    assert res["transaction_id"] == "TX-CUSTOM-1"
    assert 0.0 <= res["risk_score"] <= 1.0
    assert res["verdict"] in ["flagged", "clear"]
    assert len(res["top_factors"]) > 0

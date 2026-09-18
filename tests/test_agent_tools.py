"""
Unit tests for Agent Tool Quarantine Layer (agent/tools.py).
Verifies strict data contracts compliance, dual-mode behavior, and model-backed counterfactual inference.
"""

from agent import tools
from agent.tools import (
    counterfactual,
    get_shap_explanation,
    get_transaction,
    walk_graph,
)


class _FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_live_mode_routes_get_transaction_by_id_prefix(monkeypatch):
    """
    Regression test: live-mode get_transaction used to always hit the fraud
    engine regardless of ID, silently mis-tiering every ledger/synthetic
    lookup (caught during the integration pass — see ARCHITECTURE.md).
    It must route by ID prefix to the engine that actually owns that ID.
    """
    called_urls = []

    def fake_get(url, *args, **kwargs):
        called_urls.append(url)
        if "/ledger/transaction/" in url:
            return _FakeResponse(200, {"id": "TX-LEDGER-000001", "tier": "real_ledger"})
        if "/typology/network" in url:
            return _FakeResponse(
                200,
                {
                    "edges": [
                        {
                            "id": "TX-SYNTH-0020",
                            "tier": "synthetic_network",
                            "from_account": "ACC-A",
                            "amount": 100.0,
                            "timestamp": "2026-01-01T00:00:00Z",
                        }
                    ]
                },
            )
        if "/fraud/transaction/" in url:
            return _FakeResponse(200, {"id": "TX-CARD-1", "tier": "real_card"})
        raise AssertionError(f"unexpected URL called: {url}")

    monkeypatch.setattr(tools, "VERITY_ENV", "live")
    monkeypatch.setattr(tools.requests, "get", fake_get)

    assert get_transaction("TX-LEDGER-000001")["tier"] == "real_ledger"
    assert get_transaction("TX-SYNTH-0020")["tier"] == "synthetic_network"
    assert get_transaction("TX-CARD-1")["tier"] == "real_card"
    assert any("/ledger/transaction/" in u for u in called_urls)
    assert any("/typology/network" in u for u in called_urls)
    assert any("/fraud/transaction/" in u for u in called_urls)


def test_get_transaction_card():
    tx = get_transaction("TX-CARD-9842")
    assert tx["id"] == "TX-CARD-9842"
    assert tx["tier"] == "real_card"
    assert isinstance(tx["amount"], (int, float))
    assert tx["source_dataset"] == "creditcard.csv"


def test_get_transaction_ledger():
    tx = get_transaction("TX-LEDGER-3011")
    assert tx["id"] == "TX-LEDGER-3011"
    assert tx["tier"] == "real_ledger"
    assert tx["account_id"] == "ACC-1092"
    assert tx["amount"] == 12500.0


def test_get_transaction_synthetic():
    tx = get_transaction("TX-SYNTH-5501")
    assert tx["id"] == "TX-SYNTH-5501"
    assert tx["tier"] == "synthetic_network"
    assert tx["amount"] > 0


def test_get_shap_explanation_structure():
    exp = get_shap_explanation("TX-CARD-9842")
    assert exp["transaction_id"] == "TX-CARD-9842"
    assert 0.0 <= exp["risk_score"] <= 1.0
    assert exp["verdict"] in ["flagged", "clear"]
    assert isinstance(exp["top_factors"], list)
    assert len(exp["top_factors"]) > 0

    # Verify interpretable split rule
    interpretable_count = sum(1 for f in exp["top_factors"] if f["interpretable"])
    anonymized_count = sum(1 for f in exp["top_factors"] if not f["interpretable"])
    assert interpretable_count >= 1, "Must contain interpretable factors (Amount/Time)"
    assert anonymized_count >= 1, "Must contain anonymized features (V1-V28)"


def test_walk_graph_real_ledger():
    walk = walk_graph(account_id="ACC-1092", tier="real_ledger")
    assert walk["account_id"] == "ACC-1092"
    assert walk["tier"] == "real_ledger"
    assert isinstance(walk["steps"], list)
    assert len(walk["steps"]) > 0
    for step in walk["steps"]:
        assert step["tier"] == "real_ledger"
        assert step["from_account"] == "ACC-1092"
        assert "tool_call_id" in step
        assert "balance" in step


def test_walk_graph_synthetic_network():
    walk = walk_graph(account_id="ACC-SYN-401", tier="synthetic_network", depth=3)
    assert walk["account_id"] == "ACC-SYN-401"
    assert walk["tier"] == "synthetic_network"
    assert len(walk["steps"]) >= 2
    accounts = [s["to_account"] for s in walk["steps"]]
    assert len(set(accounts)) > 1


def test_model_backed_counterfactual_amount_reduction():
    # Reducing high amount to $50 should shift model probability and drop risk score
    result = counterfactual("TX-CARD-9842", {"Amount": 50.0})
    assert result["transaction_id"] == "TX-CARD-9842"
    assert result["original_risk_score"] >= 0.80
    assert result["recalculated_risk_score"] < 0.50
    assert result["recalculated_verdict"] == "clear"
    assert "Amount" in result["feature_attribution_deltas"]
    assert result["feature_attribution_deltas"]["Amount"] < 0.0


def test_model_backed_counterfactual_amount_increase():
    # Increasing amount should preserve high risk score
    result = counterfactual("TX-CARD-9842", {"Amount": 15000.0})
    assert result["recalculated_risk_score"] >= 0.80
    assert "Amount" in result["feature_attribution_deltas"]

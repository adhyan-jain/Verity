"""
Agent Tool Interface.
Person C: The only way the LLM touches engine data.
Exposes exactly 4 functions: get_transaction, get_shap_explanation, walk_graph, counterfactual.
"""

from typing import Dict, Any


def get_transaction(transaction_id: str) -> Dict[str, Any]:
    """
    Retrieves transaction details matching TransactionRecord contract.
    """
    # In early scaffold, can read from contracts/mock_data/ or call engine API
    # TODO: Connect to real API when available
    return {
        "id": transaction_id,
        "tier": "real_card",
        "timestamp": "2026-09-18T00:00:00Z",
        "amount": 0.0,
        "source_dataset": "creditcard.csv",
    }


def get_shap_explanation(transaction_id: str) -> Dict[str, Any]:
    """
    Retrieves SHAP factor breakdown matching FraudExplanation contract.
    """
    # TODO: Connect to real fraud engine API
    return {
        "transaction_id": transaction_id,
        "risk_score": 0.0,
        "verdict": "clear",
        "top_factors": [],
        "model_version": "v1.0",
    }


def walk_graph(
    account_id: str, tier: str = "real_ledger", depth: int = 2
) -> Dict[str, Any]:
    """
    Traverses transactions/nodes matching GraphWalkStep contract.
    tier must be 'real_ledger' or 'synthetic_network'.
    """
    # TODO: Connect to ledger or typology engine API based on tier
    return {"account_id": account_id, "tier": tier, "steps": []}


def counterfactual(
    transaction_id: str, parameter_overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Re-runs the relevant engine with modified parameters (e.g. amount or timestamp)
    and returns a fresh explanation/score rather than hallucinating answers.
    """
    # TODO: Pass overrides to fraud or typology engine and return recalculated output
    return {
        "transaction_id": transaction_id,
        "original_risk_score": 0.89,
        "recalculated_risk_score": 0.15,
        "verdict": "clear",
        "modifications": parameter_overrides,
    }

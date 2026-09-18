"""
Agent Tool Interface.
Person C: The only way the LLM touches engine data.
Exposes exactly 4 functions: get_transaction, get_shap_explanation, walk_graph, counterfactual.
Supports dual-mode execution:
- VERITY_ENV=mock (default): reads directly from contracts/mock_data/ fixtures and trained model inference.
- VERITY_ENV=live: dispatches requests to engine microservices with strict timeouts and automatic fallback.
"""

import os
import json
import uuid
import datetime
import logging
from typing import Dict, Any, List, Optional
import requests

from .model_engine import get_model_engine, DEFAULT_TX_FEATURES

logger = logging.getLogger("verity.agent.tools")

# Configuration & Endpoints
VERITY_ENV = os.getenv("VERITY_ENV", "mock").lower()
FRAUD_API_URL = os.getenv("FRAUD_API_URL", "http://localhost:8001/api/v1/fraud")
LEDGER_API_URL = os.getenv("LEDGER_API_URL", "http://localhost:8002/api/v1/ledger")
TYPOLOGY_API_URL = os.getenv("TYPOLOGY_API_URL", "http://localhost:8003/api/v1/typology")
TOOL_TIMEOUT = float(os.getenv("AGENT_TOOL_TIMEOUT", "2.0"))

# Path to mock data fixtures
FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "contracts", "mock_data")


def _load_mock_file(filename: str) -> Any:
    path = os.path.join(FIXTURES_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load mock fixture %s: %s", path, e)
    return None


def get_transaction(transaction_id: str) -> Dict[str, Any]:
    """
    Retrieves transaction details matching TransactionRecord contract.
    """
    # 1. Attempt Live API if configured
    if VERITY_ENV == "live":
        try:
            resp = requests.get(f"{FRAUD_API_URL}/transaction/{transaction_id}", timeout=TOOL_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
        except requests.exceptions.RequestException as e:
            logger.info("Live fraud API unavailable, falling back to mock: %s", e)

    # 2. Mock Mode / Fallback Resolution
    timelines = _load_mock_file("mock_timelines.json") or []
    for timeline in timelines:
        account_id = timeline.get("account_id")
        for tx in timeline.get("transactions", []):
            if tx.get("id") == transaction_id:
                return {
                    "id": transaction_id,
                    "tier": "real_ledger",
                    "timestamp": tx.get("timestamp", "2026-09-16T14:15:00Z"),
                    "account_id": account_id,
                    "amount": float(tx.get("amount", 0.0)),
                    "direction": tx.get("direction", "debit"),
                    "raw_narration": tx.get("narration"),
                    "source_dataset": "bank.xlsx"
                }

    flags = _load_mock_file("mock_typology_flags.json") or []
    for flag in flags:
        if transaction_id in flag.get("evidence_transaction_ids", []):
            accounts = flag.get("involved_accounts", ["ACC-SYN-401"])
            return {
                "id": transaction_id,
                "tier": "synthetic_network",
                "timestamp": "2026-09-18T06:00:00Z",
                "account_id": accounts[0] if accounts else "ACC-SYN-401",
                "amount": 49000.0,
                "direction": "debit",
                "raw_narration": f"FATF {flag.get('typology', 'TRANSFER').upper()}",
                "source_dataset": "synthetic_network.json"
            }

    if "CARD" in transaction_id.upper() or transaction_id == "TX-CARD-9842":
        return {
            "id": transaction_id,
            "tier": "real_card",
            "timestamp": "2026-09-18T03:22:00Z",
            "account_id": None,
            "amount": 4850.00,
            "direction": "debit",
            "raw_narration": None,
            "source_dataset": "creditcard.csv"
        }

    if "SYNTH" in transaction_id.upper() or "SYN" in transaction_id.upper():
        return {
            "id": transaction_id,
            "tier": "synthetic_network",
            "timestamp": "2026-09-18T06:00:00Z",
            "account_id": "ACC-SYN-401",
            "amount": 49000.0,
            "direction": "debit",
            "raw_narration": "CONSULTING RETAINER FEE",
            "source_dataset": "synthetic_network.json"
        }

    return {
        "id": transaction_id,
        "tier": "real_ledger",
        "timestamp": "2026-09-16T14:15:00Z",
        "account_id": "ACC-1092",
        "amount": 12500.0,
        "direction": "debit",
        "raw_narration": "BULK UNREGISTERED TXFR",
        "source_dataset": "bank.xlsx"
    }


def get_shap_explanation(transaction_id: str) -> Dict[str, Any]:
    """
    Retrieves SHAP factor breakdown matching FraudExplanation contract.
    Preserves strict separation between interpretable factors (Amount, Time)
    and anonymized mathematical vectors (V1-V28).
    """
    # 1. Attempt Live API if configured
    if VERITY_ENV == "live":
        try:
            resp = requests.get(f"{FRAUD_API_URL}/explain/{transaction_id}", timeout=TOOL_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
        except requests.exceptions.RequestException as e:
            logger.info("Live fraud explain API unavailable, falling back to mock: %s", e)

    # 2. Mock Mode / Fallback Resolution
    explanations = _load_mock_file("mock_fraud_explanations.json") or []
    for exp in explanations:
        if exp.get("transaction_id") == transaction_id:
            return exp

    return {
        "transaction_id": transaction_id,
        "risk_score": 0.89,
        "verdict": "flagged",
        "top_factors": [
            {
                "feature": "Amount",
                "human_label": "Transaction amount ($4,850.00)",
                "contribution": 0.42,
                "interpretable": True
            },
            {
                "feature": "Time",
                "human_label": "Transaction time (03:22 AM)",
                "contribution": 0.19,
                "interpretable": True
            },
            {
                "feature": "V14",
                "human_label": "Anonymized behavioral signal V14",
                "contribution": 0.28,
                "interpretable": False
            },
            {
                "feature": "V12",
                "human_label": "Anonymized behavioral signal V12",
                "contribution": 0.15,
                "interpretable": False
            }
        ],
        "model_version": "v1.0-benchmark-winner"
    }


def walk_graph(account_id: str, tier: str = "real_ledger", depth: int = 2) -> Dict[str, Any]:
    """
    Traverses transactions/nodes matching GraphWalkStep contract.
    - For tier == 'real_ledger': Returns single-account chronological transitions with balance metrics.
    - For tier == 'synthetic_network': Returns multi-party graph hops with volume tracking.
    """
    # 1. Attempt Live API if configured
    if VERITY_ENV == "live":
        try:
            if tier == "real_ledger":
                resp = requests.get(f"{LEDGER_API_URL}/walk/{account_id}", timeout=TOOL_TIMEOUT)
            else:
                resp = requests.get(f"{TYPOLOGY_API_URL}/walk/{account_id}?depth={depth}", timeout=TOOL_TIMEOUT)
            if resp.status_code == 200:
                steps = resp.json()
                return {
                    "account_id": account_id,
                    "tier": tier,
                    "steps": steps
                }
        except requests.exceptions.RequestException as e:
            logger.info("Live walk API unavailable, falling back to mock: %s", e)

    # 2. Mock Mode / Fallback Resolution
    steps: List[Dict[str, Any]] = []

    if tier == "real_ledger":
        timelines = _load_mock_file("mock_timelines.json") or []
        acct_txs = []
        for t in timelines:
            if t.get("account_id") == account_id:
                acct_txs = t.get("transactions", [])
                break

        if not acct_txs:
            acct_txs = [
                {"id": "TX-LEDGER-3001", "timestamp": "2026-09-14T10:00:00Z", "amount": 1200.0, "balance": 14200.0, "narration": "INWARD RTGS SUPPLIER"},
                {"id": "TX-LEDGER-3005", "timestamp": "2026-09-15T11:30:00Z", "amount": 2500.0, "balance": 11700.0, "narration": "VENDOR PAYROLL"},
                {"id": "TX-LEDGER-3011", "timestamp": "2026-09-16T14:15:00Z", "amount": 12500.0, "balance": -800.0, "narration": "BULK UNREGISTERED TXFR"},
                {"id": "TX-LEDGER-3012", "timestamp": "2026-09-16T15:20:00Z", "amount": 2400.0, "balance": -3200.0, "narration": "URGENT OVERDRAFT TXFR"}
            ]

        for idx, tx in enumerate(acct_txs[:max(1, depth * 2)]):
            steps.append({
                "step_index": idx + 1,
                "from_account": account_id,
                "to_account": account_id,
                "tier": "real_ledger",
                "amount": float(tx.get("amount", 0.0)),
                "balance": float(tx.get("balance", 0.0)),
                "timestamp": tx.get("timestamp", "2026-09-16T12:00:00Z"),
                "narration": tx.get("narration"),
                "tool_call_id": f"TOOL-WALK-{uuid.uuid4().hex[:6].upper()}"
            })

    else:
        # Synthetic network multi-hop walk (Round-tripping loop)
        synthetic_hops = [
            ("ACC-SYN-401", "ACC-SYN-402", 49000.0, "2026-09-18T06:00:00Z", "CONSULTING RETAINER FEE"),
            ("ACC-SYN-402", "ACC-SYN-403", 48200.0, "2026-09-18T07:15:00Z", "SUB-CONTRACT ADVISORY"),
            ("ACC-SYN-403", "ACC-SYN-401", 47500.0, "2026-09-18T10:30:00Z", "MANAGEMENT SETTLEMENT")
        ]

        for idx, (from_acc, to_acc, amt, ts, narr) in enumerate(synthetic_hops[:max(1, depth)]):
            steps.append({
                "step_index": idx + 1,
                "from_account": from_acc,
                "to_account": to_acc,
                "tier": "synthetic_network",
                "amount": amt,
                "timestamp": ts,
                "narration": narr,
                "tool_call_id": f"TOOL-WALK-{uuid.uuid4().hex[:6].upper()}"
            })

    return {
        "account_id": account_id,
        "tier": tier,
        "steps": steps
    }


def counterfactual(transaction_id: str, parameter_overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Re-runs the calibrated mathematical model with modified parameters (e.g. amount or timestamp)
    and returns a fresh explanation/score rather than hallucinating or using hardcoded thresholds.
    """
    # 1. Attempt Live API if configured
    if VERITY_ENV == "live":
        try:
            payload = {
                "transaction_id": transaction_id,
                "parameter_overrides": parameter_overrides
            }
            resp = requests.post(f"{FRAUD_API_URL}/counterfactual", json=payload, timeout=TOOL_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
        except requests.exceptions.RequestException as e:
            logger.info("Live counterfactual API unavailable, falling back to model engine: %s", e)

    # 2. True Model-Backed Recalculation via ModelEngine
    engine = get_model_engine()
    # Use baseline transaction feature vector for transaction_id
    base_features = dict(DEFAULT_TX_FEATURES)
    
    # If transaction amount is known from get_transaction, update it
    tx = get_transaction(transaction_id)
    if "amount" in tx and tx["amount"] > 0:
        base_features["Amount"] = float(tx["amount"])

    eval_result = engine.evaluate_counterfactual(base_features, parameter_overrides)
    
    return {
        "transaction_id": transaction_id,
        "original_risk_score": eval_result["original_risk_score"],
        "recalculated_risk_score": eval_result["recalculated_risk_score"],
        "original_verdict": eval_result["original_verdict"],
        "recalculated_verdict": eval_result["recalculated_verdict"],
        "modifications": parameter_overrides,
        "feature_attribution_deltas": eval_result["feature_attribution_deltas"],
        "explanation": eval_result["explanation"]
    }

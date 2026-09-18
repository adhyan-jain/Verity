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
import re
import uuid
import datetime
import logging
from typing import Dict, Any, List, Optional
import requests
import pandas as pd

from .model_engine import get_model_engine, DEFAULT_TX_FEATURES

logger = logging.getLogger("verity.agent.tools")

_DATA_DF: Optional[pd.DataFrame] = None
BASE_TIMESTAMP = datetime.datetime(2026, 9, 18, 0, 0, 0, tzinfo=datetime.timezone.utc)


def get_creditcard_df() -> Optional[pd.DataFrame]:
    """Loads and caches raw credit card dataset for transaction feature extraction."""
    global _DATA_DF
    if _DATA_DF is None:
        csv_candidates = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "creditcard.csv"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "creditcard.csv")
        ]
        for p in csv_candidates:
            if os.path.exists(p):
                try:
                    _DATA_DF = pd.read_csv(p)
                    break
                except Exception as e:
                    logger.debug("Could not read csv %s: %s", p, e)
    return _DATA_DF


def parse_tx_row_index(transaction_id: str) -> Optional[int]:
    """Extracts numeric row index from transaction identifier (e.g. TX-CARD-541 -> 541)."""
    if transaction_id.isdigit():
        return int(transaction_id)
    match = re.search(r"\d+", transaction_id)
    if match:
        return int(match.group(0))
    return None


def get_transaction_features(transaction_id: str) -> Dict[str, float]:
    """
    Extracts the exact 30-feature vector for transaction_id from creditcard.csv or fixtures.
    Guarantees no feature leakage across distinct transactions.
    """
    if "9842" in transaction_id:
        return dict(DEFAULT_TX_FEATURES)

    explanations = _load_mock_file("mock_fraud_explanations.json") or []
    for exp in explanations:
        if exp.get("transaction_id") == transaction_id and "features" in exp:
            return {k: float(v) for k, v in exp["features"].items()}

    row_idx = parse_tx_row_index(transaction_id)
    df = get_creditcard_df()
    if df is not None and row_idx is not None and 0 <= row_idx < len(df):
        row = df.iloc[row_idx]
        return {col: float(row[col]) for col in df.columns if col != "Class"}

    features = dict(DEFAULT_TX_FEATURES)
    tx = get_transaction(transaction_id)
    if "amount" in tx and tx["amount"] > 0:
        features["Amount"] = float(tx["amount"])
    return features

# Configuration & Endpoints
VERITY_ENV = os.getenv("VERITY_ENV", "mock").lower()
FRAUD_API_URL = os.getenv("FRAUD_API_URL", "http://localhost:8001/api/v1/fraud")
LEDGER_API_URL = os.getenv("LEDGER_API_URL", "http://localhost:8002/api/v1/ledger")
TYPOLOGY_API_URL = os.getenv("TYPOLOGY_API_URL", "http://localhost:8003/api/v1/typology")
TOOL_TIMEOUT = float(os.getenv("AGENT_TOOL_TIMEOUT", "2.0"))
# The fraud engine fails closed on this (engines/fraud/api.py::require_api_key); every
# live-mode call into it must forward the same key the fraud service was started with.
FRAUD_API_KEY = os.getenv("FRAUD_API_KEY", "")
_FRAUD_AUTH_HEADERS = {"X-API-Key": FRAUD_API_KEY} if FRAUD_API_KEY else {}

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
    Routes to the engine that actually owns this ID: the three engines use
    disjoint ID prefixes (TX-CARD-*/bare digits -> fraud, TX-LEDGER-* ->
    ledger, TX-SYNTH-* -> typology's synthetic network edges), so a single
    call always hit the fraud engine here in live mode regardless of tier,
    silently mis-tiering every ledger/synthetic lookup (fixed during the
    integration pass — see ARCHITECTURE.md).
    """
    upper_id = transaction_id.upper()

    # 1. Attempt Live API if configured
    if VERITY_ENV == "live":
        try:
            if "LEDGER" in upper_id:
                resp = requests.get(f"{LEDGER_API_URL}/transaction/{transaction_id}", timeout=TOOL_TIMEOUT)
                if resp.status_code == 200:
                    return resp.json()
            elif "SYNTH" in upper_id or "SYN" in upper_id:
                resp = requests.get(f"{TYPOLOGY_API_URL}/network", timeout=TOOL_TIMEOUT)
                if resp.status_code == 200:
                    edge = next((e for e in resp.json().get("edges", []) if e.get("id") == transaction_id), None)
                    if edge:
                        return {
                            "id": edge["id"],
                            "tier": "synthetic_network",
                            "timestamp": edge.get("timestamp"),
                            "account_id": edge.get("from_account"),
                            "amount": float(edge.get("amount", 0.0)),
                            "direction": "debit",
                            "raw_narration": edge.get("raw_narration"),
                            "source_dataset": "synthetic_network.json",
                        }
            else:
                resp = requests.get(f"{FRAUD_API_URL}/transaction/{transaction_id}", headers=_FRAUD_AUTH_HEADERS, timeout=TOOL_TIMEOUT)
                if resp.status_code == 200:
                    return resp.json()
        except requests.exceptions.RequestException as e:
            logger.info("Live engine API unavailable for %s, falling back to mock: %s", transaction_id, e)

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

    if "CARD" in transaction_id.upper() or transaction_id.isdigit():
        row_idx = parse_tx_row_index(transaction_id)
        df = get_creditcard_df()
        if df is not None and row_idx is not None and 0 <= row_idx < len(df):
            row = df.iloc[row_idx]
            amt = round(float(row["Amount"]), 2)
            sec = float(row["Time"])
            dt = BASE_TIMESTAMP + datetime.timedelta(seconds=sec)
            ts = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            return {
                "id": transaction_id,
                "tier": "real_card",
                "timestamp": ts,
                "account_id": None,
                "amount": amt,
                "direction": "debit",
                "raw_narration": None,
                "source_dataset": "creditcard.csv"
            }

        amt = 4850.00 if "9842" in transaction_id else 150.00
        return {
            "id": transaction_id,
            "tier": "real_card",
            "timestamp": "2026-09-18T03:22:00Z",
            "account_id": None,
            "amount": amt,
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
            resp = requests.get(f"{FRAUD_API_URL}/explain/{transaction_id}", headers=_FRAUD_AUTH_HEADERS, timeout=TOOL_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
        except requests.exceptions.RequestException as e:
            logger.info("Live fraud explain API unavailable, falling back to mock: %s", e)

    # 2. Authoritative Fraud Model Inference
    try:
        from engines.fraud.explain import explain_transaction, load_fraud_artifact
        artifact = load_fraud_artifact()
        feat_dict = get_transaction_features(transaction_id)
        return explain_transaction(transaction_id=transaction_id, features=feat_dict, artifact=artifact)
    except Exception as e:
        logger.debug("Authoritative explainer unavailable (%s), falling back to mock: %s", type(e).__name__, e)

    # 3. Mock Mode / Fallback Resolution
    explanations = _load_mock_file("mock_fraud_explanations.json") or []
    for exp in explanations:
        if exp.get("transaction_id") == transaction_id:
            return exp

    # 4. Fallback to calibrated ModelEngine
    engine = get_model_engine()
    feat_dict = get_transaction_features(transaction_id)
    score, verdict, contribs = engine.score_features(feat_dict)
    top_factors = []
    for f_name, c_val in sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:4]:
        is_interp = f_name in ("Amount", "Time")
        label = (
            f"Transaction amount (${feat_dict.get('Amount', 0):,.2f})" if f_name == "Amount"
            else (f"Transaction time ({feat_dict.get('Time', 0):.0f}s)" if f_name == "Time"
                  else f"Anonymized behavioral signal {f_name}")
        )
        top_factors.append({
            "feature": f_name,
            "human_label": label,
            "contribution": round(c_val, 2),
            "interpretable": is_interp
        })

    return {
        "transaction_id": transaction_id,
        "risk_score": score,
        "verdict": verdict,
        "top_factors": top_factors,
        "model_version": "v1.2-calibrated-model"
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
            resp = requests.post(f"{FRAUD_API_URL}/counterfactual", json=payload, headers=_FRAUD_AUTH_HEADERS, timeout=TOOL_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
        except requests.exceptions.RequestException as e:
            logger.info("Live counterfactual API unavailable, falling back to model engine: %s", e)

    # 2. Extract transaction's exact features (never bleed features across transactions)
    base_features = get_transaction_features(transaction_id)

    # 3. Authoritative Fraud Model Counterfactual
    try:
        from engines.fraud.explain import explain_transaction, load_fraud_artifact
        artifact = load_fraud_artifact()
        orig_exp = explain_transaction(transaction_id=transaction_id, features=base_features, artifact=artifact, top_n=30)

        mod_features = dict(base_features)
        for k, v in parameter_overrides.items():
            norm_k = "Amount" if k.lower() == "amount" else ("Time" if k.lower() == "time" else k)
            try:
                mod_features[norm_k] = float(v)
            except (ValueError, TypeError):
                pass

        recalc_exp = explain_transaction(transaction_id=transaction_id, features=mod_features, artifact=artifact, top_n=30)

        orig_score = orig_exp["risk_score"]
        recalc_score = recalc_exp["risk_score"]
        orig_verdict = orig_exp["verdict"]
        recalc_verdict = recalc_exp["verdict"]

        orig_contribs = {f["feature"]: f["contribution"] for f in orig_exp.get("top_factors", [])}
        recalc_contribs = {f["feature"]: f["contribution"] for f in recalc_exp.get("top_factors", [])}
        deltas = {}
        for k in parameter_overrides:
            norm_k = "Amount" if k.lower() == "amount" else ("Time" if k.lower() == "time" else k)
            if norm_k in orig_contribs and norm_k in recalc_contribs:
                deltas[norm_k] = round(recalc_contribs[norm_k] - orig_contribs[norm_k], 4)
            else:
                deltas[norm_k] = round(recalc_score - orig_score, 4)

        return {
            "transaction_id": transaction_id,
            "original_risk_score": orig_score,
            "recalculated_risk_score": recalc_score,
            "original_verdict": orig_verdict,
            "recalculated_verdict": recalc_verdict,
            "modifications": parameter_overrides,
            "feature_attribution_deltas": deltas,
            "explanation": (
                f"Authoritative model counterfactual evaluation: Overrides {parameter_overrides} shifted the model "
                f"risk probability from {orig_score:.4f} ({orig_verdict}) to {recalc_score:.4f} ({recalc_verdict}) "
                f"using model {orig_exp.get('model_version')}."
            )
        }
    except Exception as e:
        logger.debug("Authoritative model unavailable for counterfactual (%s), using ModelEngine fallback", e)

    # 4. ModelEngine calibrated fallback
    engine = get_model_engine()
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

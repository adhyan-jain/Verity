"""
FastAPI Service for Fraud Engine.
Person A: Serves get_transaction and get_shap_explanation endpoints.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

app = FastAPI(title="Verity Fraud Engine API")


@app.get("/api/v1/fraud/transaction/{transaction_id}")
def get_transaction(transaction_id: str) -> Dict[str, Any]:
    """
    Returns TransactionRecord schema for a given card fraud transaction.
    """
    # TODO: Lookup transaction in creditcard dataset
    return {
        "id": transaction_id,
        "tier": "real_card",
        "timestamp": "2026-09-18T00:00:00Z",
        "account_id": None,
        "amount": 0.0,
        "direction": "debit",
        "raw_narration": None,
        "source_dataset": "creditcard.csv"
    }


@app.get("/api/v1/fraud/explain/{transaction_id}")
def get_shap_explanation(transaction_id: str) -> Dict[str, Any]:
    """
    Returns FraudExplanation schema with SHAP attributions.
    """
    # TODO: Load model & explainer, compute SHAP factors
    return {
        "transaction_id": transaction_id,
        "risk_score": 0.0,
        "verdict": "clear",
        "top_factors": [],
        "model_version": "v1.0"
    }

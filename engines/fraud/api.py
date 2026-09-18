"""
FastAPI Service for Fraud Engine.
Person A: Serves get_transaction, get_shap_explanation, and counterfactual endpoints.
"""

import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from engines.fraud.explain import explain_transaction, load_fraud_artifact

logger = logging.getLogger("engines.fraud.api")

# Global dataset cache
_DATA_DF: pd.DataFrame | None = None
_STARTUP_ERROR: str | None = None
BASE_TIMESTAMP = datetime(2026, 9, 18, 0, 0, 0, tzinfo=timezone.utc)

FRAUD_API_KEY = os.environ.get("FRAUD_API_KEY")
_DEFAULT_CORS = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8080,http://127.0.0.1:8080,http://localhost:5173,http://127.0.0.1:5173"
_RAW_CORS = os.environ.get("FRAUD_CORS_ORIGINS") or _DEFAULT_CORS
_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in _RAW_CORS.split(",")
    if origin.strip()
]


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """
    Validates the X-API-Key header against FRAUD_API_KEY.
    If FRAUD_API_KEY is unset, the API refuses all protected requests rather
    than silently running open (fail closed, not fail open).
    """
    if not FRAUD_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Service misconfigured: FRAUD_API_KEY is not set.",
        )
    if x_api_key != FRAUD_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def require_ready() -> None:
    """Guards routes that depend on the model/dataset having loaded at startup."""
    if _STARTUP_ERROR is not None:
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready: {_STARTUP_ERROR}",
        )


def get_dataset(data_path: str = "data/raw/creditcard.csv") -> pd.DataFrame:
    """
    Loads and caches the raw credit card transactions dataframe.
    """
    global _DATA_DF
    if _DATA_DF is None:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Credit card dataset not found at {data_path}")
        _DATA_DF = pd.read_csv(data_path)
    return _DATA_DF


def parse_row_index(transaction_id: str) -> int:
    """
    Parses row index from transaction identifier (e.g. 'TX-CARD-541' -> 541, '9842' -> 9842).
    """
    if transaction_id.isdigit():
        return int(transaction_id)
    match = re.search(r"\d+", transaction_id)
    if match:
        return int(match.group(0))
    raise ValueError(f"Cannot parse transaction index from '{transaction_id}'")


def format_iso_timestamp(seconds_offset: float) -> str:
    """
    Converts elapsed seconds in dataset to an ISO8601 UTC timestamp.
    """
    dt = BASE_TIMESTAMP + timedelta(seconds=float(seconds_offset))
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# Pydantic Schemas
class TopFactor(BaseModel):
    feature: str
    human_label: str
    contribution: float
    interpretable: bool


class FraudExplanationResponse(BaseModel):
    transaction_id: str
    risk_score: float
    verdict: str
    top_factors: list[TopFactor]
    model_version: str


class TransactionRecordResponse(BaseModel):
    id: str
    tier: str = "real_card"
    timestamp: str
    account_id: str | None = None
    amount: float
    direction: str | None = "debit"
    raw_narration: str | None = None
    source_dataset: str = "creditcard.csv"


class CounterfactualRequest(BaseModel):
    transaction_id: str
    parameter_overrides: dict[str, float | int] = Field(default_factory=dict)


class CounterfactualResponse(BaseModel):
    transaction_id: str
    original_risk_score: float
    recalculated_risk_score: float
    original_verdict: str
    recalculated_verdict: str
    modifications: dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load model artifact and dataset on server boot."""
    global _STARTUP_ERROR
    try:
        load_fraud_artifact()
        get_dataset()
        _STARTUP_ERROR = None
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as e:
        _STARTUP_ERROR = str(e)
        logger.error("Startup initialization failed: %s", e)
    yield


app = FastAPI(
    title="Verity Fraud Engine API",
    description="Detection and SHAP explainability service for credit card fraud transactions",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
@app.get("/api/v1/fraud/health")
def health_check() -> dict[str, Any]:
    """Health check endpoint reporting engine status and artifact details. Unauthenticated by design for infra probes."""
    if _STARTUP_ERROR is not None:
        raise HTTPException(
            status_code=503, detail=f"Service not ready: {_STARTUP_ERROR}"
        )
    artifact = load_fraud_artifact()
    return {
        "status": "healthy",
        "engine": "fraud",
        "strategy": artifact.get(
            "strategy", "SMOTE (0.05) + LightGBM + 5-Fold Calibrated Threshold"
        ),
        "model_version": artifact.get("model_version", "v1.1-prod-calibrated"),
        "metrics": artifact.get("metrics", {}),
    }


@app.get(
    "/api/v1/fraud/transaction/{transaction_id}",
    response_model=TransactionRecordResponse,
    dependencies=[Depends(require_api_key), Depends(require_ready)],
)
def get_transaction(transaction_id: str) -> dict[str, Any]:
    """
    Returns TransactionRecord schema for a given card fraud transaction.
    """
    df = get_dataset()
    try:
        idx = parse_row_index(transaction_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if idx < 0 or idx >= len(df):
        raise HTTPException(
            status_code=404,
            detail=f"Transaction index {idx} out of range (valid range: 0 to {len(df) - 1})",
        )

    row = df.iloc[idx]
    timestamp_str = format_iso_timestamp(row["Time"])
    return {
        "id": transaction_id
        if transaction_id.startswith("TX-CARD-")
        else f"TX-CARD-{idx}",
        "tier": "real_card",
        "timestamp": timestamp_str,
        "account_id": None,
        "amount": round(float(row["Amount"]), 2),
        "direction": "debit",
        "raw_narration": None,
        "source_dataset": "creditcard.csv",
    }


@app.get(
    "/api/v1/fraud/explain/{transaction_id}",
    response_model=FraudExplanationResponse,
    dependencies=[Depends(require_api_key), Depends(require_ready)],
)
def get_shap_explanation(transaction_id: str) -> dict[str, Any]:
    """
    Returns FraudExplanation schema with SHAP attributions.
    """
    df = get_dataset()
    try:
        idx = parse_row_index(transaction_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if idx < 0 or idx >= len(df):
        raise HTTPException(
            status_code=404,
            detail=f"Transaction index {idx} out of range (valid range: 0 to {len(df) - 1})",
        )

    row = df.iloc[idx].to_dict()
    formatted_id = (
        transaction_id if transaction_id.startswith("TX-CARD-") else f"TX-CARD-{idx}"
    )
    explanation = explain_transaction(features=row, transaction_id=formatted_id)
    return explanation


@app.post(
    "/api/v1/fraud/counterfactual",
    response_model=CounterfactualResponse,
    dependencies=[Depends(require_api_key), Depends(require_ready)],
)
@app.post(
    "/api/v1/agent/counterfactual",
    response_model=CounterfactualResponse,
    dependencies=[Depends(require_api_key), Depends(require_ready)],
)
def compute_counterfactual(request: CounterfactualRequest) -> dict[str, Any]:
    """
    Recalculates risk score and verdict when transaction parameters (e.g. Amount, Time) are altered.
    """
    df = get_dataset()
    try:
        idx = parse_row_index(request.transaction_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if idx < 0 or idx >= len(df):
        raise HTTPException(
            status_code=404,
            detail=f"Transaction index {idx} out of range (valid range: 0 to {len(df) - 1})",
        )

    original_row = df.iloc[idx].to_dict()
    formatted_id = (
        request.transaction_id
        if request.transaction_id.startswith("TX-CARD-")
        else f"TX-CARD-{idx}"
    )
    orig_explanation = explain_transaction(
        features=original_row, transaction_id=formatted_id
    )

    # Apply overrides, restricted to known model features so callers can't
    # inject arbitrary keys into the feature dict passed to the model.
    allowed_params = set(original_row.keys())
    invalid_params = set(request.parameter_overrides) - allowed_params
    if invalid_params:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown parameter override(s): {sorted(invalid_params)}",
        )

    modified_row = original_row.copy()
    for param, val in request.parameter_overrides.items():
        modified_row[param] = float(val)

    # Re-compute
    recalc_explanation = explain_transaction(
        features=modified_row, transaction_id=formatted_id
    )

    return {
        "transaction_id": formatted_id,
        "original_risk_score": orig_explanation["risk_score"],
        "recalculated_risk_score": recalc_explanation["risk_score"],
        "original_verdict": orig_explanation["verdict"],
        "recalculated_verdict": recalc_explanation["verdict"],
        "modifications": request.parameter_overrides,
    }


class TransactionListItem(BaseModel):
    id: str
    tier: str = "real_card"
    timestamp: str
    amount: float
    is_ground_truth_fraud: bool


class TransactionListResponse(BaseModel):
    total: int
    offset: int
    limit: int
    transactions: list[TransactionListItem]


@app.get(
    "/api/v1/fraud/transactions",
    response_model=TransactionListResponse,
    dependencies=[Depends(require_api_key), Depends(require_ready)],
)
def list_transactions(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    flagged_only: bool = Query(default=False),
) -> dict[str, Any]:
    """
    Lists transactions with pagination and optional filter for flagged/fraud cases.
    """
    df = get_dataset()
    if flagged_only:
        filtered_df = df[df["Class"] == 1]
    else:
        filtered_df = df

    total_count = len(filtered_df)
    page_df = filtered_df.iloc[offset : offset + limit]

    results = []
    for idx, row in page_df.iterrows():
        results.append(
            {
                "id": f"TX-CARD-{idx}",
                "tier": "real_card",
                "timestamp": format_iso_timestamp(row["Time"]),
                "amount": round(float(row["Amount"]), 2),
                "is_ground_truth_fraud": bool(row["Class"] == 1),
            }
        )

    return {
        "total": total_count,
        "offset": offset,
        "limit": limit,
        "transactions": results,
    }

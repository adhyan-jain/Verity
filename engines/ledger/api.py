"""
FastAPI Service for Ledger Engine.
Person B: Serves account summaries, reconciliation anomalies, horizontal timelines,
and GraphWalkStep trajectories for the real ledger tier.
"""

import json
import os
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

try:
    from .anomalies import (
        detect_all_ledger_anomalies,
        detect_balance_breaks,
        detect_reversal_outliers,
        detect_timing_spikes,
    )
    from .parse_narrations import (
        get_all_account_summaries,
        parse_bank_ledger,
    )
    from .reconcile import compute_account_baseline
    from .timeline import TIMELINE_CACHE_FILE, build_account_timeline
except (ImportError, ValueError):
    from anomalies import (
        detect_all_ledger_anomalies,
        detect_balance_breaks,
        detect_reversal_outliers,
        detect_timing_spikes,
    )
    from parse_narrations import get_all_account_summaries, parse_bank_ledger
    from reconcile import compute_account_baseline
    from timeline import TIMELINE_CACHE_FILE, build_account_timeline

app = FastAPI(
    title="Verity Ledger Engine API",
    description="Real Ledger Account Reconciliation, Anomaly Detection & Timeline Service",
    version="1.0.0",
)

# Enable CORS for local dev dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache for parsed data
_LEDGER_DF: pd.DataFrame | None = None
_TIMELINES_CACHE: dict[str, Any] | None = None


def get_df() -> pd.DataFrame:
    global _LEDGER_DF
    if _LEDGER_DF is None:
        _LEDGER_DF = parse_bank_ledger()
    return _LEDGER_DF


def get_cached_timelines() -> dict[str, Any]:
    global _TIMELINES_CACHE
    if _TIMELINES_CACHE is None:
        if os.path.exists(TIMELINE_CACHE_FILE):
            try:
                with open(TIMELINE_CACHE_FILE, "r") as f:
                    _TIMELINES_CACHE = json.load(f)
            except (OSError, json.JSONDecodeError):
                _TIMELINES_CACHE = {}
        else:
            _TIMELINES_CACHE = {}
    return _TIMELINES_CACHE


@app.get("/api/v1/ledger/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "ledger-engine", "tier": "real_ledger"}


@app.get("/api/v1/ledger/accounts")
def list_accounts() -> list[dict[str, Any]]:
    """
    Returns summaries and high-level metrics for all 10 real bank ledger accounts.
    """
    df = get_df()
    return get_all_account_summaries(df)


@app.get("/api/v1/ledger/anomalies")
def get_all_anomalies() -> list[dict[str, Any]]:
    """
    Returns all detected ReconciliationAnomaly records across all accounts.
    """
    df = get_df()
    return detect_all_ledger_anomalies(df)


@app.get("/api/v1/ledger/anomalies/{account_id}")
def get_account_anomalies(account_id: str) -> list[dict[str, Any]]:
    """
    Returns ReconciliationAnomaly records for a specific account.
    """
    clean_acc = str(account_id).replace("'", "").strip()
    df = get_df()
    acc_df = df[df["account_id"] == clean_acc]
    if acc_df.empty:
        raise HTTPException(status_code=404, detail=f"Account {clean_acc} not found")

    baseline = compute_account_baseline(acc_df)
    breaks = detect_balance_breaks(acc_df, baseline)
    spikes = detect_timing_spikes(acc_df, baseline)
    reversals = detect_reversal_outliers(acc_df, baseline)

    anomalies = breaks + spikes + reversals
    anomalies.sort(key=lambda x: x["severity"], reverse=True)
    return anomalies


@app.get("/api/v1/ledger/timeline/{account_id}")
def get_timeline(account_id: str) -> dict[str, Any]:
    """
    Returns full visual timeline data for account_id (density curve, anomaly windows, transactions).
    """
    clean_acc = str(account_id).replace("'", "").strip()
    timelines = get_cached_timelines()

    if clean_acc in timelines:
        return timelines[clean_acc]

    df = get_df()
    acc_df = df[df["account_id"] == clean_acc]
    if acc_df.empty:
        raise HTTPException(status_code=404, detail=f"Account {clean_acc} not found")

    return build_account_timeline(clean_acc, df)


@app.get("/api/v1/ledger/walk/{account_id}")
def walk_ledger_graph(
    account_id: str, limit: int = Query(default=20, ge=1, le=100)
) -> list[dict[str, Any]]:
    """
    Returns GraphWalkStep records representing the account's own transaction sequence.
    (Real ledger tier: walk is the account's sequential transaction history).
    """
    clean_acc = str(account_id).replace("'", "").strip()
    df = get_df()
    acc_df = (
        df[df["account_id"] == clean_acc]
        .sort_values("datetime", ascending=False)
        .head(limit)
    )

    if acc_df.empty:
        raise HTTPException(status_code=404, detail=f"Account {clean_acc} not found")

    steps: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(acc_df.iterrows()):
        steps.append(
            {
                "step_index": idx + 1,
                "from_account": clean_acc
                if row["direction"] == "debit"
                else (row["counterparty"] or "EXTERNAL_ENTITY"),
                "to_account": (row["counterparty"] or "EXTERNAL_ENTITY")
                if row["direction"] == "debit"
                else clean_acc,
                "tier": "real_ledger",
                "amount": round(float(row["amount"]), 2),
                "timestamp": row["timestamp"],
                "narration": row["raw_narration"],
                "tool_call_id": f"TOOL-CALL-LEDGER-{clean_acc[-4:]}-{idx + 1}",
            }
        )
    return steps


@app.get("/api/v1/ledger/transaction/{transaction_id}")
def get_transaction(transaction_id: str) -> dict[str, Any]:
    """
    Returns TransactionRecord schema for a single ledger transaction.
    """
    df = get_df()
    match = df[df["id"] == transaction_id]
    if match.empty:
        raise HTTPException(
            status_code=404, detail=f"Transaction {transaction_id} not found"
        )

    row = match.iloc[0]
    return {
        "id": row["id"],
        "tier": "real_ledger",
        "timestamp": row["timestamp"],
        "account_id": row["account_id"],
        "amount": round(float(row["amount"]), 2),
        "direction": row["direction"],
        "raw_narration": row["raw_narration"],
        "source_dataset": "bank.xlsx",
    }


if __name__ == "__main__":
    import uvicorn

    print("Starting Ledger Engine API on http://127.0.0.1:8002 ...")
    uvicorn.run("engines.ledger.api:app", host="127.0.0.1", port=8002, reload=False)

"""
Visual Timeline Generator for Real Ledger Accounts.
Person B: Builds horizontal timeline structures with transaction density curves
and highlighted ReconciliationAnomaly bounding windows.
Resolves the sparse-data asymmetry for the analyst dashboard.
"""

import json
import os
from typing import Any

import numpy as np
import pandas as pd

try:
    from .anomalies import (
        detect_balance_breaks,
        detect_reversal_outliers,
        detect_timing_spikes,
    )
    from .parse_narrations import parse_bank_ledger
    from .reconcile import compute_account_baseline
except (ImportError, ValueError):
    from anomalies import (
        detect_balance_breaks,
        detect_reversal_outliers,
        detect_timing_spikes,
    )
    from parse_narrations import parse_bank_ledger
    from reconcile import compute_account_baseline

TIMELINE_CACHE_FILE = "data/cache/timelines.json"


def build_account_timeline(
    account_id: str,
    df: pd.DataFrame | None = None,
    max_transactions_return: int = 500,
) -> dict[str, Any]:
    """
    Generates rich, visual timeline payload for a single ledger account:
    - Account profile & baseline metrics
    - Overlaid ReconciliationAnomaly windows
    - Daily aggregated volume/density curve for time-series sparklines
    - Sampled chronological transactions including all anomaly evidence transactions
    """
    if df is None:
        df = parse_bank_ledger()

    clean_acc = str(account_id).replace("'", "").strip()
    acc_df = df[df["account_id"] == clean_acc].sort_values("datetime").copy()

    if acc_df.empty:
        return {"error": f"Account {clean_acc} not found", "account_id": clean_acc}

    baseline = compute_account_baseline(acc_df)

    # 1. Detect anomalies for this account
    breaks = detect_balance_breaks(acc_df, baseline)
    spikes = detect_timing_spikes(acc_df, baseline)
    reversals = detect_reversal_outliers(acc_df, baseline)
    anomalies = breaks + spikes + reversals
    anomalies.sort(key=lambda x: x["severity"], reverse=True)

    # Collect all evidence transaction IDs
    evidence_ids = set()
    for anom in anomalies:
        for tid in anom.get("evidence_transaction_ids", []):
            evidence_ids.add(tid)

    # 2. Compute Daily Density Curve (vectorized aggregation)
    acc_df["date_str"] = acc_df["datetime"].dt.strftime("%Y-%m-%d")
    acc_df["is_debit_num"] = np.where(
        acc_df["direction"] == "debit", acc_df["amount"], 0.0
    )
    acc_df["is_credit_num"] = np.where(
        acc_df["direction"] == "credit", acc_df["amount"], 0.0
    )

    anomaly_dates = set()
    for anom in anomalies:
        try:
            start_date = anom["window_start"][:10]
            end_date = anom["window_end"][:10]
            anomaly_dates.add(start_date)
            anomaly_dates.add(end_date)
        except (KeyError, TypeError):
            continue

    daily_agg = (
        acc_df.groupby("date_str")
        .agg(
            tx_count=("id", "count"),
            debit_volume=("is_debit_num", "sum"),
            credit_volume=("is_credit_num", "sum"),
            closing_balance=("balance", "last"),
        )
        .reset_index()
    )

    density_curve = []
    for row in daily_agg.itertuples(index=False):
        density_curve.append(
            {
                "date": row.date_str,
                "tx_count": int(row.tx_count),
                "debit_volume": round(float(row.debit_volume), 2),
                "credit_volume": round(float(row.credit_volume), 2),
                "closing_balance": round(float(row.closing_balance), 2),
                "has_anomaly": row.date_str in anomaly_dates,
            }
        )

    # 3. Transaction Stream (Preserve all evidence txns + uniform sample of normal txns)
    evidence_mask = acc_df["id"].isin(evidence_ids)
    evidence_txns = acc_df[evidence_mask]
    normal_txns = acc_df[~evidence_mask]

    remaining_slots = max(10, max_transactions_return - len(evidence_txns))
    if len(normal_txns) > remaining_slots:
        sample_step = max(1, len(normal_txns) // remaining_slots)
        sampled_normal = normal_txns.iloc[::sample_step]
    else:
        sampled_normal = normal_txns

    combined_txns = (
        pd.concat([evidence_txns, sampled_normal])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    tx_list = []
    for row in combined_txns.itertuples(index=False):
        tx_list.append(
            {
                "id": row.id,
                "tier": "real_ledger",
                "account_id": clean_acc,
                "timestamp": row.timestamp,
                "amount": round(float(row.amount), 2),
                "direction": row.direction,
                "balance": round(float(row.balance), 2),
                "payment_rail": row.payment_rail,
                "raw_narration": row.raw_narration,
                "counterparty": row.counterparty,
                "is_anomaly_evidence": row.id in evidence_ids,
            }
        )

    return {
        "account_id": clean_acc,
        "account_name": f"Commercial Corporate Account #{clean_acc[-4:]}",
        "tier": "real_ledger",
        "total_transactions": len(acc_df),
        "summary": {
            "date_start": acc_df["timestamp"].min(),
            "date_end": acc_df["timestamp"].max(),
            "initial_balance": round(float(acc_df["balance"].iloc[0]), 2),
            "final_balance": round(float(acc_df["balance"].iloc[-1]), 2),
            "min_balance": round(float(acc_df["balance"].min()), 2),
            "max_balance": round(float(acc_df["balance"].max()), 2),
            "mean_daily_velocity": round(float(baseline["velocity"]["mean_daily"]), 2),
            "anomaly_count": len(anomalies),
        },
        "anomalies": anomalies,
        "density_curve": density_curve,
        "transactions": tx_list,
    }


def precompute_all_timelines() -> dict[str, Any]:
    """
    Precomputes timelines for all 10 accounts and caches to JSON for instantaneous delivery.
    """
    df = parse_bank_ledger()
    account_ids = df["account_id"].unique().tolist()

    all_timelines = {}
    for acc in account_ids:
        print(f"Building timeline for account {acc}...")
        all_timelines[acc] = build_account_timeline(acc, df)

    os.makedirs("data/cache", exist_ok=True)
    with open(TIMELINE_CACHE_FILE, "w") as f:
        json.dump(all_timelines, f)

    print(f"Cached all {len(all_timelines)} account timelines to {TIMELINE_CACHE_FILE}")
    return all_timelines


if __name__ == "__main__":
    timelines = precompute_all_timelines()
    for acc, t in timelines.items():
        print(
            f"Account {acc} -> Total Txns: {t['total_transactions']}, Anomalies: {len(t['anomalies'])}, Density Points: {len(t['density_curve'])}"
        )

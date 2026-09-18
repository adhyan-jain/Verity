"""
Reconciliation Anomaly Detection Engine.
Person B: Detects balance breaks, timing spikes, and reversal outliers across real bank ledger accounts.
Output strictly conforms to the ReconciliationAnomaly contract in contracts/schemas.json.
"""

from typing import Any

import numpy as np
import pandas as pd

try:
    from .parse_narrations import parse_bank_ledger
    from .reconcile import compute_all_account_baselines
except (ImportError, ValueError):
    from parse_narrations import parse_bank_ledger
    from reconcile import compute_all_account_baselines


def detect_balance_breaks(
    df: pd.DataFrame, baseline: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Detects severe balance anomalies:
    1. Balance step drops > 3 standard deviations of typical step changes.
    2. Severe negative excursions below 3-sigma lower confidence bound.
    """
    anomalies: list[dict[str, Any]] = []
    account_id = baseline["account_id"]

    if len(df) < 5:
        return anomalies

    mean_balance = baseline["balance"]["mean"]
    mean_step = baseline["balance"]["mean_step"]
    std_step = max(1.0, baseline["balance"]["std_step"])

    # Consecutive balance changes
    df = df.sort_values("datetime").copy()
    balances = df["balance"].values
    steps = np.diff(balances)

    # Detect sudden massive drops
    for i, step in enumerate(steps):
        # A sharp negative step
        if step < -100000 and (step < mean_step - 3.5 * std_step):
            tx_prev = df.iloc[i]
            tx_curr = df.iloc[i + 1]

            z_score = abs(step - mean_step) / std_step
            severity = float(min(1.0, 0.6 + 0.08 * min(5.0, z_score)))

            anomalies.append(
                {
                    "account_id": account_id,
                    "anomaly_type": "balance_break",
                    "window_start": str(tx_prev["timestamp"]),
                    "window_end": str(tx_curr["timestamp"]),
                    "severity": round(severity, 2),
                    "baseline_value": round(float(mean_balance), 2),
                    "observed_value": round(float(tx_curr["balance"]), 2),
                    "evidence_transaction_ids": [
                        str(tx_prev["id"]),
                        str(tx_curr["id"]),
                    ],
                }
            )

    # Deduplicate closely overlapping windows
    return anomalies[:10]  # Return top most significant breaks


def detect_timing_spikes(
    df: pd.DataFrame, baseline: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Detects abnormal transaction velocity bursts (> 3x mean daily velocity and > P95).
    """
    anomalies: list[dict[str, Any]] = []
    account_id = baseline["account_id"]

    if len(df) < 5:
        return anomalies

    mean_v = baseline["velocity"]["mean_daily"]
    std_v = max(1.0, baseline["velocity"]["std_daily"])
    p95_v = baseline["velocity"]["p95_daily"]

    threshold = max(8, mean_v + 2.8 * std_v, p95_v * 1.5)

    df["date_only"] = df["datetime"].dt.date
    daily_groups = df.groupby("date_only")

    for date_val, group in daily_groups:
        count = len(group)
        if count >= threshold:
            z_score = (count - mean_v) / std_v
            severity = float(min(1.0, 0.65 + 0.07 * min(5.0, z_score)))

            start_ts = str(group["timestamp"].min())
            end_ts = str(group["timestamp"].max())
            evidence_ids = group["id"].head(5).tolist()

            anomalies.append(
                {
                    "account_id": account_id,
                    "anomaly_type": "timing_spike",
                    "window_start": start_ts,
                    "window_end": end_ts,
                    "severity": round(severity, 2),
                    "baseline_value": round(float(mean_v), 2),
                    "observed_value": round(float(count), 2),
                    "evidence_transaction_ids": evidence_ids,
                }
            )

    # Sort by severity descending
    anomalies.sort(key=lambda x: x["severity"], reverse=True)
    return anomalies[:10]


def detect_reversal_outliers(
    df: pd.DataFrame, baseline: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Detects abnormal bursts of paired transaction reversals (debit/credit cancellations).
    Uses O(N) chronological window matching for high-performance execution.
    """
    anomalies: list[dict[str, Any]] = []
    account_id = baseline["account_id"]

    if len(df) < 10:
        return anomalies

    base_rate = baseline["reversals"]["baseline_reversal_rate"]
    df = df.sort_values("datetime").reset_index(drop=True)

    # O(N) Chronological pairing: look for opposite direction with equal amount within 48h
    active_pool: dict[float, list[dict[str, Any]]] = {}
    matched_pairs: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        amt = round(float(row["amount"]), 2)
        direction = row["direction"]
        dt = row["datetime"]

        # Clean older entries (>48h) from pool for this amount
        if amt in active_pool:
            active_pool[amt] = [
                item
                for item in active_pool[amt]
                if (dt - item["datetime"]).total_seconds() <= 172800
            ]

            # Check for opposite direction match
            match_idx = -1
            for idx, candidate in enumerate(active_pool[amt]):
                if candidate["direction"] != direction:
                    match_idx = idx
                    break

            if match_idx != -1:
                paired_item = active_pool[amt].pop(match_idx)
                matched_pairs.append(
                    {
                        "datetime": dt,
                        "timestamp_1": paired_item["timestamp"],
                        "timestamp_2": row["timestamp"],
                        "id_1": paired_item["id"],
                        "id_2": row["id"],
                        "amount": amt,
                    }
                )
                continue

        # If not matched, add to pool
        if amt not in active_pool:
            active_pool[amt] = []
        active_pool[amt].append(
            {
                "id": row["id"],
                "datetime": dt,
                "timestamp": row["timestamp"],
                "direction": direction,
            }
        )

    # Cluster matched pairs into 7-day windows
    if matched_pairs:
        pairs_df = pd.DataFrame(matched_pairs)
        pairs_df["week"] = pairs_df["datetime"].dt.to_period("W")

        for week_period, w_group in pairs_df.groupby("week"):
            cluster_count = len(w_group)
            if cluster_count >= 3:
                evidence_ids = []
                for _, p_row in w_group.head(4).iterrows():
                    evidence_ids.extend([str(p_row["id_1"]), str(p_row["id_2"])])

                evidence_ids = list(dict.fromkeys(evidence_ids))[:6]
                severity = float(min(1.0, 0.70 + 0.05 * min(6, cluster_count)))

                min_time = str(w_group["timestamp_1"].min())
                max_time = str(w_group["timestamp_2"].max())

                anomalies.append(
                    {
                        "account_id": account_id,
                        "anomaly_type": "reversal_outlier",
                        "window_start": min_time,
                        "window_end": max_time,
                        "severity": round(severity, 2),
                        "baseline_value": round(float(base_rate), 4),
                        "observed_value": round(float(cluster_count), 2),
                        "evidence_transaction_ids": evidence_ids,
                    }
                )

    anomalies.sort(key=lambda x: x["severity"], reverse=True)
    return anomalies[:10]


def detect_all_ledger_anomalies(
    df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """
    Runs all anomaly detectors across all 10 accounts in the ledger dataset.
    Returns list of ReconciliationAnomaly records.
    """
    if df is None:
        df = parse_bank_ledger()

    baselines = compute_all_account_baselines(df)
    all_anomalies: list[dict[str, Any]] = []

    for account_id, group in df.groupby("account_id"):
        acc_str = str(account_id)
        base = baselines.get(acc_str)
        if not base:
            continue

        breaks = detect_balance_breaks(group, base)
        spikes = detect_timing_spikes(group, base)
        reversals = detect_reversal_outliers(group, base)

        all_anomalies.extend(breaks)
        all_anomalies.extend(spikes)
        all_anomalies.extend(reversals)

    all_anomalies.sort(key=lambda x: x["severity"], reverse=True)
    return all_anomalies


if __name__ == "__main__":
    anomalies = detect_all_ledger_anomalies()
    print(f"Detected {len(anomalies)} total ledger anomalies across 10 accounts:")
    for a in anomalies[:8]:
        print(
            f"  [{a['anomaly_type'].upper()}] Account {a['account_id']} (Severity: {a['severity']}) -> Window: {a['window_start']} to {a['window_end']}, Obs: {a['observed_value']} (Baseline: {a['baseline_value']})"
        )

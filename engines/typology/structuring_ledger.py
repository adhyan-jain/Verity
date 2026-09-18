"""
Step 3 — Collective Anomaly Detector: Structuring / Smurfing on bank.xlsx.

Detects when an account accumulates multiple transactions whose rolling window
sum falls just below a round reporting threshold — the classic "structuring"
(smurfing) pattern used to evade currency transaction reporting.

Algorithm:
  For each account, compute a rolling sum of transaction amounts over a
  configurable time window (default: 30 days).  Flag any window where the sum
  falls in (threshold - margin, threshold).

Output is INDEPENDENT from the point-anomaly score (Step 1) and classifier
score (Step 2).  Each flag is a separate row in the structuring table.

Configurable parameters:
  threshold   : reporting threshold in currency units (default 1000.0)
  margin      : look-below margin, i.e. flag if sum in (threshold-margin, threshold)
                (default 100.0 → flags sums in [900, 1000))
  window_days : rolling window duration in calendar days (default 30)
  min_txns    : minimum number of transactions in the window to flag (default 2)
                (avoids flagging single large transactions just below threshold)
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from typing import Any

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)


def detect_structuring(
    df: pd.DataFrame | None = None,
    threshold: float = 1_000.0,
    margin: float = 100.0,
    window_days: int = 30,
    min_txns: int = 2,
) -> list[dict[str, Any]]:
    """
    Detects structuring (smurfing) patterns in bank.xlsx ledger.

    A flag is raised for every rolling window where:
        (threshold - margin) <= rolling_sum < threshold
        AND the window contains >= min_txns transactions.

    Parameters
    ----------
    df          : Parsed ledger DataFrame. If None, loaded from disk.
    threshold   : Currency reporting threshold (default $1,000).
    margin      : Detection margin below threshold (default $100).
    window_days : Rolling window in days (default 30).
    min_txns    : Minimum transactions in flagged window (default 2).

    Returns
    -------
    List of structuring flag dicts, sorted by severity descending.
    Each dict conforms to:
        account_id, window_start, window_end, window_sum, txn_count,
        threshold, margin, gap_to_threshold, severity, evidence_ids
    """
    if df is None:
        from engines.ledger.parse_narrations import parse_bank_ledger
        df = parse_bank_ledger()

    lower_bound = threshold - margin
    flags: list[dict[str, Any]] = []

    for account_id, acct_df in df.groupby("account_id"):
        # Work only with debits (the reporting obligation is triggered by outflows)
        acct = (
            acct_df[acct_df["direction"] == "debit"]
            .sort_values("datetime")
            .reset_index(drop=True)
        )
        if acct.empty:
            continue

        window_delta = timedelta(days=window_days)

        # Sliding window: for each transaction, look back window_days
        # O(N) with a two-pointer approach
        left = 0
        window_sum = 0.0
        window_ids: list[str] = []
        window_amts: list[float] = []

        for right in range(len(acct)):
            row = acct.iloc[right]
            window_sum += float(row["amount"])
            window_ids.append(str(row["id"]))
            window_amts.append(float(row["amount"]))

            # Advance left pointer to maintain window_days
            while (
                left < right
                and (row["datetime"] - acct.iloc[left]["datetime"]) > window_delta
            ):
                window_sum -= float(acct.iloc[left]["amount"])
                window_ids.pop(0)
                window_amts.pop(0)
                left += 1

            txn_count = right - left + 1

            # Flag condition
            if lower_bound <= window_sum < threshold and txn_count >= min_txns:
                gap = threshold - window_sum
                # Severity: closer to threshold = more suspicious
                severity = round(1.0 - (gap / margin), 4)
                severity = max(0.0, min(1.0, severity))

                flags.append({
                    "account_id":       str(account_id),
                    "window_start":     str(acct.iloc[left]["timestamp"]),
                    "window_end":       str(row["timestamp"]),
                    "window_sum":       round(window_sum, 2),
                    "txn_count":        txn_count,
                    "threshold":        threshold,
                    "margin":           margin,
                    "gap_to_threshold": round(gap, 2),
                    "severity":         severity,
                    "evidence_ids":     list(window_ids[:8]),   # cap at 8
                })

    flags.sort(key=lambda x: x["severity"], reverse=True)
    return flags


def structuring_summary(flags: list[dict[str, Any]]) -> dict[str, Any]:
    """Returns aggregate counts for the structuring detector output."""
    if not flags:
        return {"total_flags": 0, "affected_accounts": 0, "max_severity": 0.0}

    df = pd.DataFrame(flags)
    return {
        "total_flags":        len(flags),
        "affected_accounts":  df["account_id"].nunique(),
        "max_severity":       round(float(df["severity"].max()), 4),
        "mean_severity":      round(float(df["severity"].mean()), 4),
        "mean_window_sum":    round(float(df["window_sum"].mean()), 2),
        "mean_txn_count":     round(float(df["txn_count"].mean()), 2),
        "accounts_flagged":   df["account_id"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    print("Running structuring/smurfing detector on bank.xlsx ...")
    flags = detect_structuring(threshold=1_000.0, margin=100.0, window_days=30)
    summary = structuring_summary(flags)
    print(f"Structuring flags: {summary['total_flags']} across "
          f"{summary['affected_accounts']} accounts")
    print(f"Max severity: {summary['max_severity']:.4f}  "
          f"Mean severity: {summary['mean_severity']:.4f}")
    print("\nTop 10 flags:")
    for f in flags[:10]:
        print(
            f"  [{f['account_id']}] {f['window_start']} -> {f['window_end']} "
            f"sum={f['window_sum']:,.2f} ({f['txn_count']} txns) "
            f"gap={f['gap_to_threshold']:.2f} sev={f['severity']:.4f}"
        )


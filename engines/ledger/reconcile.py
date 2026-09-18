"""
Account Reconciliation & Statistical Baseline Engine.
Person B: Calculates running baselines and per-account operational statistics
to detect deviations in velocity, balance breaks, and reversal frequencies.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
try:
    from .parse_narrations import parse_bank_ledger, get_account_transactions
except (ImportError, ValueError):
    from parse_narrations import parse_bank_ledger, get_account_transactions



def compute_account_baseline(account_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes comprehensive statistical baselines for a single account's transaction history.
    """
    if account_df.empty:
        return {}

    account_id = str(account_df["account_id"].iloc[0])
    df = account_df.sort_values("datetime").copy()
    
    # 1. Transaction Volumes & Counts
    total_txns = len(df)
    debits = df[df["direction"] == "debit"]
    credits = df[df["direction"] == "credit"]
    
    # 2. Daily Velocity Metrics
    df["date_only"] = df["datetime"].dt.date
    daily_counts = df.groupby("date_only").size()
    
    mean_daily_velocity = float(daily_counts.mean()) if not daily_counts.empty else 0.0
    std_daily_velocity = float(daily_counts.std()) if len(daily_counts) > 1 else 0.0
    p95_daily_velocity = float(np.percentile(daily_counts, 95)) if not daily_counts.empty else mean_daily_velocity
    max_daily_velocity = int(daily_counts.max()) if not daily_counts.empty else 0

    # 3. Amount Metrics
    debit_amounts = debits["amount"].values if not debits.empty else np.array([0.0])
    credit_amounts = credits["amount"].values if not credits.empty else np.array([0.0])
    
    debit_stats = {
        "count": int(len(debits)),
        "total": float(debits["amount"].sum()),
        "mean": float(np.mean(debit_amounts)),
        "std": float(np.std(debit_amounts)),
        "median": float(np.median(debit_amounts)),
        "p95": float(np.percentile(debit_amounts, 95))
    }
    
    credit_stats = {
        "count": int(len(credits)),
        "total": float(credits["amount"].sum()),
        "mean": float(np.mean(credit_amounts)),
        "std": float(np.std(credit_amounts)),
        "median": float(np.median(credit_amounts)),
        "p95": float(np.percentile(credit_amounts, 95))
    }

    # 4. Balance Trajectory Metrics
    balances = df["balance"].values
    mean_balance = float(np.mean(balances))
    std_balance = float(np.std(balances))
    min_balance = float(np.min(balances))
    max_balance = float(np.max(balances))
    
    # Balance step change distribution (diff between consecutive reported balances)
    balance_diffs = np.diff(balances) if len(balances) > 1 else np.array([0.0])
    mean_balance_step = float(np.mean(balance_diffs))
    std_balance_step = float(np.std(balance_diffs))

    # 5. Timing Interval Metrics (in hours)
    if len(df) > 1:
        time_deltas = (df["datetime"].diff().dt.total_seconds() / 3600.0).dropna()
        mean_inter_arrival_hours = float(time_deltas.mean())
        std_inter_arrival_hours = float(time_deltas.std())
    else:
        mean_inter_arrival_hours = 0.0
        std_inter_arrival_hours = 0.0

    # 6. Reversal Rate Baseline (O(N) sliding pool)
    reversal_matches = 0
    pool: Dict[float, List[Any]] = {}
    for _, row in df.iterrows():
        amt = round(float(row["amount"]), 2)
        direction = row["direction"]
        dt = row["datetime"]
        if amt in pool:
            pool[amt] = [item for item in pool[amt] if (dt - item["datetime"]).total_seconds() <= 172800]
            match_idx = -1
            for idx, candidate in enumerate(pool[amt]):
                if candidate["direction"] != direction:
                    match_idx = idx
                    break
            if match_idx != -1:
                pool[amt].pop(match_idx)
                reversal_matches += 1
                continue
        if amt not in pool:
            pool[amt] = []
        pool[amt].append({"datetime": dt, "direction": direction})

    baseline_reversal_rate = float(reversal_matches / max(1, total_txns))


    return {
        "account_id": account_id,
        "total_transactions": total_txns,
        "date_range": {
            "start": df["timestamp"].min(),
            "end": df["timestamp"].max()
        },
        "velocity": {
            "mean_daily": mean_daily_velocity,
            "std_daily": std_daily_velocity,
            "p95_daily": p95_daily_velocity,
            "max_daily": max_daily_velocity,
            "mean_interval_hours": mean_inter_arrival_hours,
            "std_interval_hours": std_inter_arrival_hours
        },
        "debits": debit_stats,
        "credits": credit_stats,
        "balance": {
            "mean": mean_balance,
            "std": std_balance,
            "min": min_balance,
            "max": max_balance,
            "mean_step": mean_balance_step,
            "std_step": std_balance_step,
            "lower_bound_3sigma": mean_balance - 3 * std_balance
        },
        "reversals": {
            "matched_reversal_count": reversal_matches,
            "baseline_reversal_rate": baseline_reversal_rate
        }
    }


def compute_all_account_baselines(df: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, Any]]:
    """
    Computes and returns a dictionary of baselines for all accounts in the ledger.
    """
    if df is None:
        df = parse_bank_ledger()
    
    baselines: Dict[str, Dict[str, Any]] = {}
    for account_id, group in df.groupby("account_id"):
        baselines[str(account_id)] = compute_account_baseline(group)
    return baselines


if __name__ == "__main__":
    df = parse_bank_ledger()
    baselines = compute_all_account_baselines(df)
    for acc_id, b in baselines.items():
        print(f"Account {acc_id} -> Daily Vel: {b['velocity']['mean_daily']:.1f} txns/day (P95: {b['velocity']['p95_daily']:.1f}), Reversal Rate: {b['reversals']['baseline_reversal_rate']:.3f}")

"""
Visual Timeline Generator for Real Ledger Accounts.
Person B: Builds per-account horizontal strip of transactions with highlighted anomaly windows.
"""

from typing import List, Dict, Any
import pandas as pd


def build_account_timeline(account_id: str, df: pd.DataFrame, anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Builds data structure for frontend timeline view (resolves the sparse-data asymmetry).
    """
    # TODO: Extract chronologically sorted transactions for account_id and overlay anomaly windows
    return {
        "account_id": account_id,
        "anomalies": anomalies,
        "transactions": []
    }

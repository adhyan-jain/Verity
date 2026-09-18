"""
Narration Parser & Transaction Normalizer for Bank Ledger Data.
Person B: Ingests bank.xlsx (116,201 rows across 10 accounts), cleans data,
extracts payment channels, counterparties, and normalizes into canonical TransactionRecord format.
Includes high-speed Parquet cache for sub-50ms subsequent loads.
"""

import os
import re
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

CACHE_DIR = "data/cache"
PARQUET_CACHE_FILE = os.path.join(CACHE_DIR, "bank_parsed.parquet")


def extract_payment_rail(narration: str) -> str:
    """
    Classifies transaction payment rail / channel from narration text.
    """
    if not isinstance(narration, str):
        return "OTHER"
    
    narration_upper = narration.upper()
    
    if "RTGS" in narration_upper:
        return "RTGS"
    elif "NEFT" in narration_upper:
        return "NEFT"
    elif "UPI" in narration_upper:
        return "UPI"
    elif "IMPS" in narration_upper:
        return "IMPS"
    elif "INDO GIBL" in narration_upper:
        return "INDO_GIBL"
    elif "FDRL" in narration_upper or "INTERNAL" in narration_upper:
        return "INTERNAL_TRANSFER"
    elif "TRF" in narration_upper or "TRANSFER" in narration_upper:
        return "BOOK_TRANSFER"
    elif "CHQ" in narration_upper or "CHEQUE" in narration_upper or "CLEARING" in narration_upper:
        return "CHEQUE"
    elif "CASH" in narration_upper or "CDM" in narration_upper or "ATM" in narration_upper:
        return "CASH_ATM"
    elif "POS" in narration_upper or "ECOM" in narration_upper:
        return "CARD_POS"
    else:
        return "GENERAL_TRANSFER"


def extract_counterparty(narration: str) -> Optional[str]:
    """
    Extracts counterparty entity or narration tag from raw bank narration.
    """
    if not isinstance(narration, str) or not narration.strip():
        return None
    
    cleaned = narration.strip()
    # Common prefixes to strip
    cleaned = re.sub(r'^(TRF (?:FROM|FRM|TO)|NEFT|RTGS|UPI/CR|UPI/DR|FDRL/)\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned if cleaned else narration.strip()


def parse_bank_ledger(file_path: str = "data/raw/bank.xlsx", force_reload: bool = False) -> pd.DataFrame:
    """
    Parses and structures bank.xlsx into standardized schema.
    Uses Parquet cache for instantaneous reloads.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    if not force_reload and os.path.exists(PARQUET_CACHE_FILE):
        return pd.read_parquet(PARQUET_CACHE_FILE)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw ledger file not found at: {file_path}")
    
    print(f"Parsing raw ledger file: {file_path} (this may take a few seconds on first load)...")
    raw_df = pd.read_excel(file_path, sheet_name=0)
    
    # 1. Clean Account No
    account_series = raw_df["Account No"].astype(str).str.replace("'", "").str.strip()
    
    # 2. Parse Dates
    dates = pd.to_datetime(raw_df["DATE"], errors="coerce")
    value_dates = pd.to_datetime(raw_df["VALUE DATE"], errors="coerce").fillna(dates)
    
    # 3. Handle Amounts & Direction
    withdrawal_amt = pd.to_numeric(raw_df["WITHDRAWAL AMT"], errors="coerce").fillna(0.0)
    deposit_amt = pd.to_numeric(raw_df["DEPOSIT AMT"], errors="coerce").fillna(0.0)
    balance_amt = pd.to_numeric(raw_df["BALANCE AMT"], errors="coerce").fillna(0.0)
    
    is_debit = withdrawal_amt > 0
    amount = np.where(is_debit, withdrawal_amt, deposit_amt)
    direction = np.where(is_debit, "debit", "credit")
    
    # 4. Narrations & Rails
    raw_narrations = raw_df["TRANSACTION DETAILS"].fillna("").astype(str)
    payment_rails = [extract_payment_rail(n) for n in raw_narrations]
    counterparties = [extract_counterparty(n) for n in raw_narrations]
    
    chq_series = raw_df["CHQ.NO."].fillna("").astype(str).str.replace(".0", "", regex=False).str.strip()
    chq_series = np.where(chq_series.isin(["", "nan", "None"]), None, chq_series)
    
    # 5. Assemble normalized DataFrame
    parsed_df = pd.DataFrame({
        "id": [f"TX-LEDGER-{i+1:06d}" for i in range(len(raw_df))],
        "tier": "real_ledger",
        "account_id": account_series,
        "timestamp": dates.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "datetime": dates,
        "value_datetime": value_dates,
        "amount": amount.astype(float),
        "direction": direction,
        "balance": balance_amt.astype(float),
        "raw_narration": raw_narrations,
        "payment_rail": payment_rails,
        "counterparty": counterparties,
        "chq_no": chq_series,
        "source_dataset": "bank.xlsx"
    })
    
    # Sort chronologically per account
    parsed_df = parsed_df.sort_values(["account_id", "datetime"]).reset_index(drop=True)
    # Re-assign sequential IDs after sorting
    parsed_df["id"] = [f"TX-LEDGER-{i+1:06d}" for i in range(len(parsed_df))]
    
    # Save cache
    parsed_df.to_parquet(PARQUET_CACHE_FILE, index=False)
    print(f"Successfully cached {len(parsed_df)} parsed transactions to {PARQUET_CACHE_FILE}")
    
    return parsed_df


def get_account_transactions(account_id: str, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Returns filtered transactions for a specific account.
    """
    if df is None:
        df = parse_bank_ledger()
    clean_acc = str(account_id).replace("'", "").strip()
    return df[df["account_id"] == clean_acc].copy()


def get_all_account_summaries(df: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
    """
    Returns high-level statistics for all 10 accounts.
    """
    if df is None:
        df = parse_bank_ledger()
    
    summaries = []
    for acc, group in df.groupby("account_id"):
        debits = group[group["direction"] == "debit"]
        credits = group[group["direction"] == "credit"]
        summaries.append({
            "account_id": acc,
            "total_transactions": int(len(group)),
            "start_date": group["timestamp"].min(),
            "end_date": group["timestamp"].max(),
            "initial_balance": float(group["balance"].iloc[0]),
            "final_balance": float(group["balance"].iloc[-1]),
            "min_balance": float(group["balance"].min()),
            "max_balance": float(group["balance"].max()),
            "total_debit_volume": float(debits["amount"].sum()),
            "total_credit_volume": float(credits["amount"].sum()),
            "debit_count": int(len(debits)),
            "credit_count": int(len(credits)),
            "top_rail": group["payment_rail"].mode().iloc[0] if not group["payment_rail"].empty else "OTHER"
        })
    return summaries


if __name__ == "__main__":
    df = parse_bank_ledger()
    print("Parsed bank dataset summary:")
    summaries = get_all_account_summaries(df)
    for s in summaries:
        print(f"  Account {s['account_id']}: {s['total_transactions']} txns, Balance Range: [{s['min_balance']:,.2f} to {s['max_balance']:,.2f}]")

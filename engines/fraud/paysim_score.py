"""
PaySim Scoring — applies trained paysim_model.pkl to bank.xlsx transactions
and returns the flagged queue ranked by risk score.

The bank.xlsx ledger has no orig/dest balance structure (it is a single-account
ledger, not a bipartite transfer graph).  We therefore construct PaySim-compatible
proxy features from what IS available:

    balance_drain_ratio   → amount / opening_balance_of_the_day
    tx_velocity           → log1p(row_index within account, chronological)
    orig_dest_mismatch    → |balance_change - amount| (direction-adjusted)
    amount_log            → log1p(amount)
    orig_open_log         → log1p(balance_before_tx)
    orig_close_zero       → 1 if balance after ≈ 0
    type_* dummies        → mapped from payment_rail:
                              NEFT/RTGS/IMPS → TRANSFER
                              CASH/ATM       → CASH_OUT (debit) / CASH_IN (credit)
                              UPI/POS/ECOM   → PAYMENT
                              CHQ/GENERAL    → PAYMENT

Returns:
    list[dict] — one row per transaction, keys:
        id, account_id, timestamp, amount, direction, payment_rail,
        raw_narration, risk_score_lgb, risk_score_rf, risk_score,
        flagged (bool), conformal_lo, conformal_hi  (None until Step 7 wired)
"""

from __future__ import annotations

import os
import pickle
from typing import Any

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

PAYSIM_MODEL_PATH = os.path.join(_HERE, "paysim_model.pkl")
_ARTIFACT: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Type mapping from bank.xlsx payment_rail → PaySim type
# ---------------------------------------------------------------------------
_RAIL_TO_TYPE: dict[str, str] = {
    "NEFT":              "TRANSFER",
    "RTGS":              "TRANSFER",
    "IMPS":              "TRANSFER",
    "INDO_GIBL":         "TRANSFER",
    "INTERNAL_TRANSFER": "TRANSFER",
    "BOOK_TRANSFER":     "TRANSFER",
    "CASH_ATM":          "CASH_OUT",   # direction adjusted below
    "CHEQUE":            "PAYMENT",
    "CARD_POS":          "PAYMENT",
    "UPI":               "PAYMENT",
    "GENERAL_TRANSFER":  "PAYMENT",
    "OTHER":             "PAYMENT",
}


def _load_artifact() -> dict[str, Any]:
    global _ARTIFACT
    if _ARTIFACT is None:
        if not os.path.exists(PAYSIM_MODEL_PATH):
            raise FileNotFoundError(
                f"paysim_model.pkl not found at {PAYSIM_MODEL_PATH}. "
                "Run engines/fraud/paysim_train.py first."
            )
        with open(PAYSIM_MODEL_PATH, "rb") as f:
            _ARTIFACT = pickle.load(f)
    return _ARTIFACT


def _bank_to_paysim_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs PaySim-compatible feature matrix from bank.xlsx parsed dataframe.
    Expects columns: amount, balance, direction, payment_rail,
                     datetime (sorted chronologically per account).
    """
    out = pd.DataFrame(index=df.index)

    # 1. balance_drain_ratio: amount / balance_before_tx
    #    balance column is CLOSING balance, so opening ≈ balance + amount (debit)
    #    or balance - amount (credit)
    is_debit = df["direction"] == "debit"
    opening_balance = np.where(
        is_debit,
        df["balance"] + df["amount"],  # before debit
        df["balance"] - df["amount"],  # before credit
    )
    opening_balance = np.clip(opening_balance, 1e-2, None)
    out["balance_drain_ratio"] = (df["amount"] / opening_balance).clip(0, 1)

    # 2. tx_velocity: log1p of intra-account chronological row index
    out["tx_velocity"] = np.log1p(
        df.groupby("account_id").cumcount()
    )

    # 3. orig_dest_mismatch: |balance_change - amount|
    #    For a real ledger with a single account there is no dest account,
    #    so we use the residual between the amount and actual balance change.
    balance_change = df["balance"].diff().fillna(0).abs()
    out["orig_dest_mismatch"] = np.log1p((balance_change - df["amount"]).abs())

    # 4. amount_log
    out["amount_log"] = np.log1p(df["amount"])

    # 5. orig_open_log
    out["orig_open_log"] = np.log1p(np.clip(opening_balance, 0, None))

    # 6. orig_close_zero — balance after tx is ≈ 0
    out["orig_close_zero"] = (df["balance"].abs() < 1.0).astype(float)

    # 7. Type dummies
    # Map rail → PaySim type, then adjust CASH_ATM by direction
    paysim_type = df["payment_rail"].map(_RAIL_TO_TYPE).fillna("PAYMENT")
    # Debits from cash/ATM → CASH_OUT; credits → CASH_IN
    paysim_type = paysim_type.where(
        ~((paysim_type == "CASH_OUT") & (~is_debit)), "CASH_IN"
    )

    for col in ["type_PAYMENT", "type_TRANSFER", "type_CASH_OUT",
                "type_CASH_IN", "type_DEBIT"]:
        type_key = col.replace("type_", "")
        out[col] = (paysim_type == type_key).astype(float)

    return out.fillna(0.0)


def score_bank_ledger(
    df: pd.DataFrame | None = None,
    threshold_lgb: float = 0.5,
    threshold_rf: float = 0.5,
    ensemble_weight_lgb: float = 0.6,
) -> list[dict[str, Any]]:
    """
    Scores all bank.xlsx transactions with the trained PaySim models.

    Parameters
    ----------
    df : pre-loaded parsed ledger dataframe (from parse_bank_ledger()).
         If None, loads from disk.
    threshold_lgb / threshold_rf : classification thresholds (default 0.5).
    ensemble_weight_lgb : weight for LightGBM in ensemble risk score (0–1).

    Returns
    -------
    List of dicts, one per transaction, sorted by risk_score descending.
    """
    artifact = _load_artifact()
    lgb_model = artifact["lgb_model"]
    rf_model  = artifact["rf_model"]

    if df is None:
        import sys
        sys.path.insert(0, _ROOT)
        from engines.ledger.parse_narrations import parse_bank_ledger
        df = parse_bank_ledger()

    # Build feature matrix
    X = _bank_to_paysim_features(df)

    # Align columns to training schema
    for col in artifact["feature_names"]:
        if col not in X.columns:
            X[col] = 0.0
    X = X[artifact["feature_names"]]

    lgb_proba = lgb_model.predict_proba(X)[:, 1]
    rf_proba  = rf_model.predict_proba(X)[:, 1]

    ensemble = (
        ensemble_weight_lgb * lgb_proba
        + (1 - ensemble_weight_lgb) * rf_proba
    )

    flagged_mask = (lgb_proba >= threshold_lgb) | (rf_proba >= threshold_rf)

    records = []
    for idx, (i, row) in enumerate(df.iterrows()):
        records.append({
            "id":               row["id"],
            "account_id":       row["account_id"],
            "timestamp":        row["timestamp"],
            "amount":           float(row["amount"]),
            "direction":        row["direction"],
            "payment_rail":     row["payment_rail"],
            "raw_narration":    row.get("raw_narration", ""),
            "risk_score_lgb":   round(float(lgb_proba[idx]), 4),
            "risk_score_rf":    round(float(rf_proba[idx]), 4),
            "risk_score":       round(float(ensemble[idx]), 4),
            "flagged":          bool(flagged_mask[idx]),
            "conformal_lo":     None,  # filled by Step 7
            "conformal_hi":     None,
        })

    records.sort(key=lambda x: x["risk_score"], reverse=True)
    return records


def get_flagged_queue(
    df: pd.DataFrame | None = None,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """
    Returns only the flagged transactions, optionally limited to top_n.
    """
    all_records = score_bank_ledger(df)
    flagged = [r for r in all_records if r["flagged"]]
    if top_n is not None:
        flagged = flagged[:top_n]
    return flagged


if __name__ == "__main__":
    queue = get_flagged_queue()
    print(f"Flagged transactions in bank.xlsx: {len(queue)}")
    for r in queue[:10]:
        print(
            f"  [{r['account_id']}] {r['id']} amt={r['amount']:,.2f} "
            f"risk={r['risk_score']:.3f} lgb={r['risk_score_lgb']:.3f} "
            f"rf={r['risk_score_rf']:.3f} rail={r['payment_rail']}"
        )

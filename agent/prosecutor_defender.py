"""
Step 5 — Prosecutor / Defender Agent Pair.

Before a flagged case is shown as "confirmed," this module runs two
structured analytical passes:

  PROSECUTOR — builds the strongest evidence-backed case for fraud:
    - Score magnitude and percentile
    - PaySim feature contributions (which features drove the score)
    - Balance-drain severity
    - Structuring flags if present
    - Velocity burst indicators

  DEFENDER — actively searches for innocent explanations:
    - Recurring payment match (same amount ± 5%, same rail, prior history)
    - Known merchant / payee pattern (same counterparty seen frequently)
    - Seasonal pattern (same calendar period in prior years shows similar volume)
    - Balance restore (credited back within N days after the flagged debit)

  RESOLUTION:
    - If the defender's explanation cites a REAL matching prior transaction or
      statistical pattern in the data, the flag is downgraded or cleared.
    - Final verdict: confirmed | downgraded | cleared
    - Both arguments are always logged (demoable).

No LLM is required for this module — all reasoning is rule-based and
evidence-grounded.  The LLM can be layered on top via agent/loop.py.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("verity.agent.prosecutor_defender")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

LOG_PATH = os.path.join(_ROOT, "logs", "pd_log.jsonl")

# Thresholds
RECURRING_AMT_TOL  = 0.05   # ±5% amount match for recurring detection
RECURRING_MIN_HITS = 2      # min prior occurrences to call it "recurring"
SEASONAL_WINDOW    = 30     # days around same calendar date in prior years
RESTORE_WINDOW_DAYS = 7     # days to look for credit restore after flagged debit
SCORE_DOWNGRADE_THRESHOLD = 0.5   # defender explanation downgrades if score < this
DEFENDER_GROUNDED_THRESHOLD = 0   # at least 1 grounded reason = grounded explanation


@dataclass
class ProsecutorVerdict:
    account_id: str
    transaction_id: str
    risk_score: float
    arguments: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class DefenderVerdict:
    account_id: str
    transaction_id: str
    arguments: list[str] = field(default_factory=list)
    grounded_reasons: list[str] = field(default_factory=list)
    cited_transaction_ids: list[str] = field(default_factory=list)
    is_grounded: bool = False   # True if ≥ 1 reason backed by real data


@dataclass
class CaseResolution:
    account_id: str
    transaction_id: str
    original_score: float
    final_score: float
    verdict: str          # "confirmed" | "downgraded" | "cleared"
    prosecutor: ProsecutorVerdict
    defender: DefenderVerdict
    resolution_reason: str


# ---------------------------------------------------------------------------
# Prosecutor
# ---------------------------------------------------------------------------

def _run_prosecutor(
    tx_row: pd.Series,
    acct_df: pd.DataFrame,
    risk_score: float,
    all_scores: list[float],
    structuring_flags: list[dict[str, Any]] | None = None,
) -> ProsecutorVerdict:
    """Builds the prosecution case for a single flagged transaction."""
    acc_id = str(tx_row["account_id"])
    tx_id  = str(tx_row["id"])
    arguments: list[str] = []
    evidence_ids: list[str] = [tx_id]

    # 1. Score percentile
    percentile = float(np.mean(np.array(all_scores) <= risk_score)) * 100
    arguments.append(
        f"Risk score {risk_score:.3f} is at the {percentile:.0f}th percentile "
        f"across all scored transactions — higher than "
        f"{percentile:.0f}% of the portfolio."
    )

    # 2. Amount vs account median
    acct_amounts = acct_df["amount"].values
    median_amt  = float(np.median(acct_amounts))
    mad_amt     = float(np.median(np.abs(acct_amounts - median_amt)))
    tx_amount   = float(tx_row["amount"])
    if mad_amt > 0:
        z_score = abs(tx_amount - median_amt) / max(mad_amt, 1e-2)
        if z_score > 2.0:
            arguments.append(
                f"Transaction amount {tx_amount:,.2f} is {z_score:.1f}× MAD "
                f"above account median {median_amt:,.2f} — a statistical outlier."
            )

    # 3. Balance drain
    is_debit = tx_row["direction"] == "debit"
    if is_debit:
        balance_after = float(tx_row["balance"])
        opening_balance = balance_after + tx_amount
        drain_ratio = tx_amount / max(opening_balance, 1)
        if drain_ratio > 0.6:
            arguments.append(
                f"Balance-drain ratio is {drain_ratio:.2%} — this single "
                f"transaction consumed {drain_ratio:.2%} of the account's "
                f"opening balance."
            )

    # 4. Velocity burst: transactions on same date
    tx_date = pd.to_datetime(tx_row["datetime"]).date()
    same_day = acct_df[pd.to_datetime(acct_df["datetime"]).dt.date == tx_date]
    if len(same_day) > 3:
        arguments.append(
            f"Transaction occurred on a day with {len(same_day)} total "
            f"transactions for this account — an elevated velocity burst."
        )
        evidence_ids.extend(same_day["id"].head(5).tolist())

    # 5. Structuring cluster
    if structuring_flags:
        acct_flags = [f for f in structuring_flags if f["account_id"] == acc_id]
        if acct_flags:
            top = acct_flags[0]
            arguments.append(
                f"Account is flagged for structuring: rolling sum "
                f"{top['window_sum']:,.2f} falls {top['gap_to_threshold']:.2f} "
                f"below reporting threshold {top['threshold']:,.2f} "
                f"(severity {top['severity']:.3f})."
            )
            evidence_ids.extend(top.get("evidence_ids", [])[:3])

    if not arguments:
        arguments.append(
            f"Classifier flagged this transaction with risk score {risk_score:.3f}."
        )

    return ProsecutorVerdict(
        account_id=acc_id,
        transaction_id=tx_id,
        risk_score=risk_score,
        arguments=arguments,
        evidence_ids=list(dict.fromkeys(evidence_ids)),
    )


# ---------------------------------------------------------------------------
# Defender
# ---------------------------------------------------------------------------

def _run_defender(
    tx_row: pd.Series,
    acct_df: pd.DataFrame,
) -> DefenderVerdict:
    """Searches for innocent explanations, citing real prior transactions."""
    acc_id = str(tx_row["account_id"])
    tx_id  = str(tx_row["id"])
    arguments: list[str] = []
    grounded_reasons: list[str] = []
    cited_ids: list[str] = []

    tx_amount   = float(tx_row["amount"])
    tx_datetime = pd.to_datetime(tx_row["datetime"])
    tx_rail     = str(tx_row.get("payment_rail", ""))
    tx_narration = str(tx_row.get("raw_narration", ""))

    # Look at prior transactions only (before flagged tx)
    prior = acct_df[pd.to_datetime(acct_df["datetime"]) < tx_datetime].copy()

    # 1. Recurring payment match
    if not prior.empty:
        amt_lo = tx_amount * (1 - RECURRING_AMT_TOL)
        amt_hi = tx_amount * (1 + RECURRING_AMT_TOL)
        same_rail_prior = prior[
            (prior["amount"] >= amt_lo)
            & (prior["amount"] <= amt_hi)
            & (prior["payment_rail"] == tx_rail)
        ]
        if len(same_rail_prior) >= RECURRING_MIN_HITS:
            hit_ids = same_rail_prior["id"].tolist()[:5]
            reason = (
                f"Amount {tx_amount:,.2f} on rail {tx_rail} matches "
                f"{len(same_rail_prior)} prior transactions "
                f"(e.g. {hit_ids[0]}) — consistent with a recurring payment."
            )
            arguments.append(reason)
            grounded_reasons.append(reason)
            cited_ids.extend(hit_ids)
        else:
            arguments.append(
                f"No recurring payment match found for amount {tx_amount:,.2f} "
                f"on rail {tx_rail} in prior history."
            )

    # 2. Known counterparty / merchant pattern
    if tx_narration and not prior.empty:
        # Extract first meaningful word from narration as proxy counterparty
        words = [w for w in tx_narration.split() if len(w) > 3]
        if words:
            keyword = words[0].upper()
            # Search narrations
            matches = prior[
                prior["raw_narration"].fillna("").str.upper().str.contains(keyword, regex=False)
            ]
            if len(matches) >= 1:
                hit_ids = matches["id"].tolist()[:3]
                reason = (
                    f"Narration keyword '{keyword}' appears in "
                    f"{len(matches)} prior transaction(s) "
                    f"(e.g. {hit_ids[0]}) — known counterparty."
                )
                arguments.append(reason)
                grounded_reasons.append(reason)
                cited_ids.extend(hit_ids)
            else:
                arguments.append(
                    f"Counterparty keyword '{keyword}' has no prior appearance "
                    f"in account history — first-time counterparty."
                )

    # 3. Seasonal pattern (same ±SEASONAL_WINDOW days in prior years)
    if not prior.empty:
        tx_month = tx_datetime.month
        tx_day   = tx_datetime.day
        # Look for transactions in the same month/day window across prior years
        prior["_dt"] = pd.to_datetime(prior["datetime"])
        seasonal = prior[
            (prior["_dt"].dt.month == tx_month)
            & (
                (prior["_dt"].dt.day - tx_day).abs() <= 7  # ±7 days same month
            )
            & (prior["_dt"].dt.year < tx_datetime.year)
        ]
        if len(seasonal) >= 1:
            hit_ids = seasonal["id"].tolist()[:3]
            amt_vals = seasonal["amount"].values
            reason = (
                f"Seasonal match: {len(seasonal)} transaction(s) in prior "
                f"year(s) around this calendar date "
                f"(avg amount: {float(np.mean(amt_vals)):,.2f}). "
                f"Example: {hit_ids[0]}."
            )
            arguments.append(reason)
            grounded_reasons.append(reason)
            cited_ids.extend(hit_ids)
        else:
            arguments.append(
                "No seasonal pattern found — this timing has no precedent in "
                "prior years for this account."
            )
        prior.drop(columns=["_dt"], inplace=True)

    # 4. Balance restore (credit within RESTORE_WINDOW_DAYS after debit)
    if tx_row.get("direction") == "debit" and not prior.empty:
        restore_window_end = tx_datetime + timedelta(days=RESTORE_WINDOW_DAYS)
        future_credits = acct_df[
            (pd.to_datetime(acct_df["datetime"]) > tx_datetime)
            & (pd.to_datetime(acct_df["datetime"]) <= restore_window_end)
            & (acct_df["direction"] == "credit")
            & (acct_df["amount"] >= tx_amount * 0.9)
        ]
        if not future_credits.empty:
            hit_ids = future_credits["id"].tolist()[:2]
            reason = (
                f"A credit of ≥{tx_amount*0.9:,.2f} was received within "
                f"{RESTORE_WINDOW_DAYS} days after this debit "
                f"(tx {hit_ids[0]}) — possible salary/refund restore."
            )
            arguments.append(reason)
            grounded_reasons.append(reason)
            cited_ids.extend(hit_ids)

    is_grounded = len(grounded_reasons) >= DEFENDER_GROUNDED_THRESHOLD + 1

    return DefenderVerdict(
        account_id=acc_id,
        transaction_id=tx_id,
        arguments=arguments,
        grounded_reasons=grounded_reasons,
        cited_transaction_ids=list(dict.fromkeys(cited_ids)),
        is_grounded=is_grounded,
    )


# ---------------------------------------------------------------------------
# Resolution logic
# ---------------------------------------------------------------------------

def _resolve(
    prosecutor: ProsecutorVerdict,
    defender: DefenderVerdict,
) -> CaseResolution:
    original_score = prosecutor.risk_score
    n_grounded = len(defender.grounded_reasons)

    if n_grounded == 0:
        # No grounded defence → confirm
        verdict = "confirmed"
        final_score = original_score
        reason = "Defender found no grounded innocent explanation."
    elif n_grounded == 1 and original_score >= SCORE_DOWNGRADE_THRESHOLD:
        # One weak ground → downgrade
        verdict = "downgraded"
        final_score = max(0.0, original_score - 0.15)
        reason = (
            f"Defender grounded 1 innocent explanation "
            f"({defender.grounded_reasons[0][:80]}). "
            f"Score reduced from {original_score:.3f} to {final_score:.3f}."
        )
    elif n_grounded >= 2:
        # Strong defence → clear if score was borderline, else downgrade
        if original_score < SCORE_DOWNGRADE_THRESHOLD + 0.15:
            verdict = "cleared"
            final_score = 0.0
            reason = (
                f"Defender grounded {n_grounded} innocent explanations. "
                f"Flag cleared."
            )
        else:
            verdict = "downgraded"
            final_score = max(0.0, original_score - 0.25)
            reason = (
                f"Defender grounded {n_grounded} explanations; score high "
                f"enough to retain as suspicious. Score: {final_score:.3f}."
            )
    else:
        verdict = "downgraded"
        final_score = max(0.0, original_score - 0.10)
        reason = f"Partial defence: {n_grounded} grounded reason(s)."

    return CaseResolution(
        account_id=prosecutor.account_id,
        transaction_id=prosecutor.transaction_id,
        original_score=original_score,
        final_score=round(final_score, 4),
        verdict=verdict,
        prosecutor=prosecutor,
        defender=defender,
        resolution_reason=reason,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def adjudicate_case(
    tx_row: pd.Series,
    acct_df: pd.DataFrame,
    risk_score: float,
    all_scores: list[float],
    structuring_flags: list[dict[str, Any]] | None = None,
    log: bool = True,
) -> CaseResolution:
    """
    Full prosecutor/defender pass for a single flagged transaction.

    Parameters
    ----------
    tx_row          : DataFrame row for the flagged transaction.
    acct_df         : All transactions for that account.
    risk_score      : Classifier risk score for this transaction.
    all_scores      : Portfolio-wide scores (for percentile computation).
    structuring_flags: Output from detect_structuring() for context.
    log             : If True, appends result to LOG_PATH.

    Returns
    -------
    CaseResolution dataclass with verdict and both argument sets.
    """
    prosecutor = _run_prosecutor(
        tx_row, acct_df, risk_score, all_scores, structuring_flags
    )
    defender = _run_defender(tx_row, acct_df)
    resolution = _resolve(prosecutor, defender)

    if log:
        _log_resolution(resolution)

    return resolution


def adjudicate_queue(
    flagged_records: list[dict[str, Any]],
    bank_df: pd.DataFrame | None = None,
    structuring_flags: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Runs prosecutor/defender on every record in the flagged queue.

    Parameters
    ----------
    flagged_records : Output from paysim_score.get_flagged_queue().
    bank_df         : Parsed bank ledger (loaded if None).
    structuring_flags: Output from structuring_ledger.detect_structuring().

    Returns
    -------
    List of serialised CaseResolution dicts, sorted by final_score descending.
    """
    if bank_df is None:
        import sys
        sys.path.insert(0, _ROOT)
        from engines.ledger.parse_narrations import parse_bank_ledger
        bank_df = parse_bank_ledger()

    all_scores = [r["risk_score"] for r in flagged_records]
    resolutions = []

    for record in flagged_records:
        tx_id  = record["id"]
        acc_id = record["account_id"]
        acct_df = bank_df[bank_df["account_id"] == acc_id]

        # Find the matching row
        tx_rows = bank_df[bank_df["id"] == tx_id]
        if tx_rows.empty:
            continue
        tx_row = tx_rows.iloc[0]

        resolution = adjudicate_case(
            tx_row=tx_row,
            acct_df=acct_df,
            risk_score=record["risk_score"],
            all_scores=all_scores,
            structuring_flags=structuring_flags,
            log=True,
        )
        resolutions.append(_resolution_to_dict(resolution))

    resolutions.sort(key=lambda x: x["final_score"], reverse=True)
    return resolutions


def _resolution_to_dict(r: CaseResolution) -> dict[str, Any]:
    return {
        "account_id":        r.account_id,
        "transaction_id":    r.transaction_id,
        "original_score":    r.original_score,
        "final_score":       r.final_score,
        "verdict":           r.verdict,
        "resolution_reason": r.resolution_reason,
        "prosecutor": {
            "arguments":   r.prosecutor.arguments,
            "evidence_ids": r.prosecutor.evidence_ids,
        },
        "defender": {
            "arguments":         r.defender.arguments,
            "grounded_reasons":  r.defender.grounded_reasons,
            "cited_ids":         r.defender.cited_transaction_ids,
            "is_grounded":       r.defender.is_grounded,
        },
    }


def _log_resolution(r: CaseResolution) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    record = _resolution_to_dict(r)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import sys
    sys.path.insert(0, _ROOT)
    from engines.ledger.parse_narrations import parse_bank_ledger
    from engines.typology.structuring_ledger import detect_structuring

    print("Loading bank ledger ...")
    bank_df = parse_bank_ledger()

    # Use a sample flagged record for demo
    sample_tx = bank_df.iloc[100]
    acct_df = bank_df[bank_df["account_id"] == sample_tx["account_id"]]
    all_scores = [0.3] * 1000 + [0.75]  # mock score distribution

    struct_flags = detect_structuring(bank_df)
    resolution = adjudicate_case(
        tx_row=sample_tx,
        acct_df=acct_df,
        risk_score=0.75,
        all_scores=all_scores,
        structuring_flags=struct_flags,
    )

    d = _resolution_to_dict(resolution)
    print(f"\n=== Verdict: {d['verdict'].upper()} ===")
    print(f"Score: {d['original_score']:.3f} -> {d['final_score']:.3f}")
    print(f"Resolution: {d['resolution_reason']}")
    print("\nPROSECUTOR ARGUMENTS:")
    for a in d["prosecutor"]["arguments"]:
        print(f"  • {a}")
    print("\nDEFENDER ARGUMENTS:")
    for a in d["defender"]["arguments"]:
        prefix = "✓" if a in d["defender"]["grounded_reasons"] else "✗"
        print(f"  {prefix} {a}")
    print(f"\nLogged to: {LOG_PATH}")


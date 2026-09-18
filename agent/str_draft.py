"""
Deterministic Suspicious Transaction Report (STR) draft builder.

Consumes an already-completed agent investigation (trace_events + narrative,
the same `Case`-shaped data `agent/loop.py` returns) and produces:

1. Per-sentence citations, re-validated INDEPENDENTLY against the existing
   grounding rules (`agent/grounding.py`) — this module never trusts that the
   caller's narrative was already grounded upstream. That's what makes this a
   real second check rather than a relabeled first one.
2. Validator notes for any sentence that fails the check.
3. An FIU-IND-shaped structured draft: account/transaction identifiers, date
   range, cited amounts, suggested "reasons for suspicion" categories, and an
   explicit list of what a real STR needs that this system cannot supply.

100% deterministic — no LLM call anywhere in this module. See
docs/STR_FEASIBILITY.md §1.3/§1.4 for why: the narrative is generated
upstream by the agent loop (deterministic by default, optionally LLM-backed);
this module only re-validates and reformats data that already exists as
structured JSON, and exact regex/string matching is strictly more reliable
and reproducible here than a second LLM call would be. Nothing in this
module is persisted — every call recomputes from the request payload, so a
re-run investigation always produces a fresh, in-sync draft.
"""

import re
from datetime import datetime, timezone
from typing import Any

from .grounding import audit_grounding

ACCOUNT_ID_PATTERN = re.compile(r"\bACC-[A-Za-z0-9\-]+\b")
TRANSACTION_ID_PATTERN = re.compile(r"\bTX-[A-Za-z0-9\-]+\b")
AMOUNT_PATTERN = re.compile(r"\$-?[\d,]+(?:\.\d{1,2})?")

# True regardless of case content: this system has no KYC/customer-identity
# layer and no bank-side compliance configuration anywhere in it.
STATIC_MISSING_SECTIONS: list[str] = [
    "Part 2 - Principal Officer details (bank compliance-team identity/contact): not configured in this system.",
    "Part 3 - Reporting Branch details (branch name, BSR code, address): not configured in this system.",
    "Part 4 / Annexure A - Individual(s) linked to the transaction (name, customer ID, address): "
    "NOT AVAILABLE. Verity has no KYC/customer-identity data anywhere in its pipeline.",
    "Part 5 / Annexure B - Legal person(s)/entity(ies) linked (name, registration, related individuals): "
    "NOT AVAILABLE for the same reason.",
    "Part 8 - Details of action taken / prior investigation status: requires compliance-team input, "
    "not derivable from detection-engine data.",
]

REASON_CODE_LABELS: dict[str, str] = {
    "C": "Multiple accounts",
    "D": "Activity in account",
    "E": "Nature of transaction",
    "F": "Value of transaction",
}

HIGH_RISK_THRESHOLD = 0.75


def _extract_unique(pattern: re.Pattern[str], *texts: str | None) -> list[str]:
    """Collects every distinct regex match across `texts`, preserving first-seen order."""
    found: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for match in pattern.findall(text or ""):
            if match not in seen:
                seen.add(match)
                found.append(match)
    return found


def _evidence_texts(trace_events: list[dict[str, Any]]) -> list[str]:
    parts: list[str] = []
    for evt in trace_events:
        parts.append(evt.get("tool_output_summary", ""))
        parts.append(evt.get("narration_sentence", ""))
        parts.append(str(evt.get("tool_input", {})))
    return parts


def _suggest_reason_codes(
    tier_origin: str, trace_events: list[dict[str, Any]], risk_score: float
) -> list[dict[str, str]]:
    """
    Deterministic, heuristic suggestions only — an analyst must confirm these,
    they are not an authoritative classification. Mirrors the actual FIU-IND
    banking STR form's "Reasons for suspicion" categories (fiuindia.gov.in);
    "A Identity of client" / "B Background of client" are never suggested —
    this system has no identity data to base that judgment on.
    """
    tools_called = {evt.get("tool_called") for evt in trace_events}
    suggestions: list[dict[str, str]] = []

    if tier_origin == "real_ledger" and "walk_graph" in tools_called:
        suggestions.append(
            {
                "code": "D",
                "label": REASON_CODE_LABELS["D"],
                "basis": "Ledger anomaly detected in account activity (balance break / timing spike / reversal outlier).",
            }
        )
    if tier_origin == "synthetic_network" and "walk_graph" in tools_called:
        suggestions.append(
            {
                "code": "E",
                "label": REASON_CODE_LABELS["E"],
                "basis": "FATF typology pattern (structuring / round-tripping / rapid layering) detected across the transaction network.",
            }
        )
        suggestions.append(
            {
                "code": "C",
                "label": REASON_CODE_LABELS["C"],
                "basis": "Multiple linked accounts observed during the graph walk.",
            }
        )
    if risk_score >= HIGH_RISK_THRESHOLD:
        suggestions.append(
            {
                "code": "F",
                "label": REASON_CODE_LABELS["F"],
                "basis": f"Elevated risk score ({risk_score:.2f}) relative to the engine's calibrated threshold.",
            }
        )

    return suggestions


def build_str_draft(
    case_id: str,
    tier_origin: str,
    primary_transaction_id: str,
    risk_score: float,
    trace_events: list[dict[str, Any]],
    narrative: str,
) -> dict[str, Any]:
    """
    Builds the full STR draft: grounded+cited sentences, validator notes for
    rejected sentences, and the FIU-IND-shaped exportable_data structure.
    Pure function, no I/O, no persistence — safe to call on every click.
    """
    audit = audit_grounding(trace_events, narrative or "")

    sentences_with_citations: list[dict[str, Any]] = []
    for sentence in audit["retained_sentences"]:
        event_id = audit["supporting_event_ids"].get(sentence)
        cited_event = next((e for e in trace_events if e.get("event_id") == event_id), None)
        sentences_with_citations.append(
            {
                "sentence": sentence,
                "event_id": event_id,
                "tool_called": cited_event.get("tool_called") if cited_event else None,
                "timestamp": cited_event.get("timestamp") if cited_event else None,
            }
        )

    validator_notes: list[dict[str, str]] = [
        {"sentence": sentence, "reason": audit["rejection_reasons"].get(sentence, "Failed independent grounding re-verification.")}
        for sentence in audit["pruned_sentences"]
    ]

    evidence_texts = _evidence_texts(trace_events)
    account_ids = _extract_unique(ACCOUNT_ID_PATTERN, *evidence_texts)
    transaction_ids = _extract_unique(TRANSACTION_ID_PATTERN, *evidence_texts, primary_transaction_id)
    amounts_cited = _extract_unique(AMOUNT_PATTERN, *evidence_texts)

    timestamps = sorted({evt["timestamp"] for evt in trace_events if evt.get("timestamp")})
    date_range = {"start": timestamps[0] if timestamps else None, "end": timestamps[-1] if timestamps else None}

    missing_sections = list(STATIC_MISSING_SECTIONS)
    if tier_origin == "real_card":
        missing_sections.insert(
            0,
            "This case is a point-in-time card-fraud score, not an AML behavioral pattern. An STR is "
            "fundamentally an AML instrument tied to a customer relationship — treat this draft as "
            "illustrative only for the card-fraud tier, not a genuine STR candidate.",
        )
    if tier_origin == "synthetic_network":
        missing_sections.insert(
            0,
            "Every account and transaction referenced below comes from Verity's SYNTHETIC typology "
            "network, not a real customer relationship. Do not present any name or account number "
            "from this tier as real.",
        )
    if not account_ids:
        missing_sections.append(
            "Part 6 / Annexure C - Account(s) linked to the transaction: no account identifier was "
            "present in the investigation evidence for this case."
        )

    exportable_data = {
        "case_id": case_id,
        "tier_origin": tier_origin,
        "primary_transaction_id": primary_transaction_id,
        "risk_score": risk_score,
        "account_ids": account_ids,
        "transaction_ids": transaction_ids,
        "amounts_cited": amounts_cited,
        "date_range": date_range,
        "reasons_for_suspicion": _suggest_reason_codes(tier_origin, trace_events, risk_score),
        "missing_sections": missing_sections,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    return {
        "narrative": " ".join(item["sentence"] for item in sentences_with_citations),
        "sentences_with_citations": sentences_with_citations,
        "validator_notes": validator_notes,
        "exportable_data": exportable_data,
    }

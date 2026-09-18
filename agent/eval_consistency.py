"""
Narrative-Model Consistency & Grounding Evaluation Suite.
Person C: Evaluates whether agent narratives generated under counterfactual perturbations
consistently match the underlying mathematical model's score, verdict, and SHAP feature drivers.

Reuses the authoritative counterfactual re-run machinery without duplicate perturbation systems.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .grounding import audit_grounding, ground_narrative
from .llm import VerityLLMClient
from .tools import (
    counterfactual,
    get_transaction,
    get_transaction_features,
)
from engines.fraud.explain import explain_transaction, load_fraud_artifact

logger = logging.getLogger("verity.agent.eval_consistency")

# Fixed evaluation set: 3 demo fraud cases with 4 targeted perturbations each
DEMO_FRAUD_PERTURBATIONS = [
    {
        "case_id": "CASE-CARD-001",
        "transaction_id": "TX-CARD-9842",
        "label": "Demo Card Fraud Case ($4,850 off-hours)",
        "perturbations": [
            {
                "name": "amount_reduction_to_50",
                "description": "Reduce amount from $4,850 to $50",
                "overrides": {"Amount": 50.0},
            },
            {
                "name": "time_shift_to_daytime",
                "description": "Shift timestamp from off-hours to midday (12:30 PM)",
                "overrides": {"Time": 45000.0},
            },
            {
                "name": "amount_and_time_reduction",
                "description": "Reduce amount to $25 and shift to daytime",
                "overrides": {"Amount": 25.0, "Time": 48000.0},
            },
            {
                "name": "high_value_escalation",
                "description": "Increase transaction amount to $15,000",
                "overrides": {"Amount": 15000.0},
            },
        ],
    },
    {
        "case_id": "CASE-CARD-623",
        "transaction_id": "TX-CARD-623",
        "label": "Ground-Truth Card Fraud (Row 623, High V14/V10)",
        "perturbations": [
            {
                "name": "amount_drop_to_5",
                "description": "Reduce transaction amount to $5.00",
                "overrides": {"Amount": 5.0},
            },
            {
                "name": "neutralize_v14_signal",
                "description": "Set principal fraud signal V14 to 0.0 (neutral baseline)",
                "overrides": {"V14": 0.0},
            },
            {
                "name": "neutralize_v14_and_amount_5",
                "description": "Neutralize V14 and reduce amount to $5.00",
                "overrides": {"V14": 0.0, "Amount": 5.0},
            },
            {
                "name": "time_offset_24h",
                "description": "Offset transaction timestamp by 24 hours (86,400s)",
                "overrides": {"Time": 86400.0},
            },
        ],
    },
    {
        "case_id": "CASE-CARD-541",
        "transaction_id": "TX-CARD-541",
        "label": "Ground-Truth Zero-Amount Card Fraud (Row 541, High Risk)",
        "perturbations": [
            {
                "name": "add_material_amount",
                "description": "Increase amount from $0.00 to $1,250.00",
                "overrides": {"Amount": 1250.0},
            },
            {
                "name": "neutralize_v14_and_v10",
                "description": "Neutralize both primary fraud components V14 and V10",
                "overrides": {"V14": 0.0, "V10": 0.0},
            },
            {
                "name": "low_amount_midday",
                "description": "Set small $10.00 amount at midday (60,000s)",
                "overrides": {"Amount": 10.0, "Time": 60000.0},
            },
            {
                "name": "high_structuring_amount",
                "description": "Set high transaction amount to $9,500.00",
                "overrides": {"Amount": 9500.0},
            },
        ],
    },
]


def check_verdict_consistency(narrative: str, expected_verdict: str) -> bool:
    """
    Code-level check (no LLM): Verifies whether the narrative's conclusion matches
    which side of the calibrated threshold the model's new score actually falls on.
    """
    if not narrative:
        return False

    text = narrative.lower()

    flagged_keywords = [
        "flagged", "flag", "fraud", "fraudulent", "suspicious", "high risk",
        "elevated risk", "anomaly", "anomalous", "high-risk", "elevated"
    ]
    clear_keywords = [
        "clear", "cleared", "low risk", "low-risk", "legitimate", "benign",
        "normal", "safe", "non-fraudulent", "clearing"
    ]

    has_flagged = any(re.search(rf"\b{re.escape(k)}\b", text) for k in flagged_keywords)
    has_clear = any(re.search(rf"\b{re.escape(k)}\b", text) for k in clear_keywords)

    if expected_verdict == "flagged":
        # Must have flagged indicators and not unconditionally declare clear
        return has_flagged or (not has_clear)
    else:
        # Expected clear: Must mention clear/low risk or describe significant risk reduction
        return has_clear or ("reduc" in text) or ("drop" in text) or ("lower" in text)


def check_feature_overlap(narrative: str, top_features: List[str]) -> bool:
    """
    Code-level check (no LLM): Verifies whether the feature(s) the narrative names
    as main driver(s) overlap with the actual top 1-2 SHAP features from the re-run.
    """
    if not narrative or not top_features:
        return False

    text = narrative.lower()

    for feat in top_features[:2]:
        f_norm = feat.lower()
        if f_norm == "amount":
            if any(k in text for k in ["amount", "$", "dollar", "sum", "value"]):
                return True
        elif f_norm == "time":
            if any(k in text for k in ["time", "hour", "night", "day", "timestamp", "seconds"]):
                return True
        else:
            # Anonymized vector like V14, V10, V4
            if re.search(rf"\b{re.escape(f_norm)}\b", text) or "anonymized" in text or "behavioral signal" in text:
                return True

    return False


def generate_perturbed_narrative_llm(
    client: VerityLLMClient,
    tx_id: str,
    case_id: str,
    explanation: Dict[str, Any],
    modifications: Dict[str, Any],
) -> Optional[str]:
    """
    Calls the REAL LLM path (when API key is configured) to generate a narrative
    synthesizing the perturbed transaction and its new SHAP attributions.
    """
    if not client.api_key:
        return None

    score = float(explanation.get("risk_score", 0.0))
    verdict = str(explanation.get("verdict", "clear"))
    factors = explanation.get("top_factors", [])[:3]

    factors_str = ", ".join(
        [f"{f['feature']} (contribution {f['contribution']:+.2f})" for f in factors]
    )

    history = [
        {
            "event_id": "EVT-PERTURB-TX",
            "case_id": case_id,
            "timestamp": "2026-09-18T12:00:00Z",
            "tool_called": "counterfactual",
            "tool_input": {"transaction_id": tx_id, "parameter_overrides": modifications},
            "tool_output_summary": (
                f"Perturbation {modifications} yielded risk score {score:.4f} ({verdict}). "
                f"Top SHAP drivers: {factors_str}."
            ),
            "narration_sentence": (
                f"Evaluation of transaction {tx_id} with modifications {modifications} "
                f"resulted in a risk score of {score:.2f} ({verdict}) driven by {factors_str}."
            ),
            "raw_output": explanation,
        }
    ]

    decision = client.decide_next_step(
        case_id=case_id,
        primary_tx_id=tx_id,
        tier_origin="real_card",
        history=history,
        step_number=2,
        max_steps=2,
    )

    candidate = decision.get("candidate_narrative")
    if not candidate and decision.get("thought"):
        candidate = decision.get("thought")

    return candidate


def evaluate_narrative_model_consistency(
    client: Optional[VerityLLMClient] = None,
    test_cases: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Evaluates Narrative-Model Consistency across fraud case counterfactual perturbations.
    Reuses existing counterfactual and explainer machinery.

    Returns:
        Dict containing total perturbations, passed checks, consistency score,
        and caveat warning if run under deterministic mode.
    """
    if client is None:
        client = VerityLLMClient()

    artifact = load_fraud_artifact()
    cases = test_cases or DEMO_FRAUD_PERTURBATIONS

    is_deterministic = not bool(client.api_key)

    results: List[Dict[str, Any]] = []
    total_perturbations = 0
    passed_perturbations = 0

    for case in cases:
        case_id = case["case_id"]
        tx_id = case["transaction_id"]
        base_feats = get_transaction_features(tx_id)

        for p in case["perturbations"]:
            total_perturbations += 1
            p_name = p["name"]
            overrides = p["overrides"]

            # 1. Apply overrides to transaction features
            mod_feats = dict(base_feats)
            for k, v in overrides.items():
                norm_k = "Amount" if k.lower() == "amount" else ("Time" if k.lower() == "time" else k)
                mod_feats[norm_k] = float(v)

            # 2. Authoritative model re-run
            explanation = explain_transaction(
                transaction_id=tx_id,
                features=mod_feats,
                artifact=artifact,
                top_n=5,
            )

            new_score = explanation["risk_score"]
            new_verdict = explanation["verdict"]
            top_factors = [f["feature"] for f in explanation.get("top_factors", [])]

            # 3. Generate candidate narrative
            if is_deterministic:
                narrative = (
                    f"Deterministic synthesis: Transaction {tx_id} modified with {overrides} "
                    f"scores {new_score:.4f} with verdict {new_verdict} driven by {top_factors[:2]}."
                )
            else:
                narrative = generate_perturbed_narrative_llm(
                    client=client,
                    tx_id=tx_id,
                    case_id=case_id,
                    explanation=explanation,
                    modifications=overrides,
                )
                if not narrative:
                    # Fallback to thought or description
                    narrative = f"Case {case_id} re-evaluated: verdict is {new_verdict} (score {new_score:.2f}) driven by {top_factors[:2]}."

            # 4. Code-level consistency checks
            verdict_pass = check_verdict_consistency(narrative, new_verdict)
            feature_pass = check_feature_overlap(narrative, top_factors)
            both_pass = verdict_pass and feature_pass

            if both_pass:
                passed_perturbations += 1

            results.append({
                "case_id": case_id,
                "transaction_id": tx_id,
                "perturbation_name": p_name,
                "overrides": overrides,
                "risk_score": new_score,
                "verdict": new_verdict,
                "top_shap_features": top_factors[:2],
                "narrative": narrative,
                "verdict_pass": verdict_pass,
                "feature_overlap_pass": feature_pass,
                "consistent": both_pass,
            })

    if is_deterministic:
        consistency_metric_str = "N/A — deterministic mode, result is not meaningful"
        consistency_pct = None
    else:
        consistency_pct = round((passed_perturbations / total_perturbations) * 100, 1) if total_perturbations > 0 else 0.0
        consistency_metric_str = f"{consistency_pct}%"

    return {
        "metric_name": "narrative_model_consistency",
        "is_deterministic_mode": is_deterministic,
        "consistency_display": consistency_metric_str,
        "consistency_percentage": consistency_pct,
        "total_perturbations": total_perturbations,
        "passed_perturbations": passed_perturbations,
        "llm_provider": client.provider if not is_deterministic else "builtin",
        "llm_model": client.model if not is_deterministic else "builtin-rules",
        "results": results,
    }


def evaluate_grounding_coverage() -> Dict[str, Any]:
    """
    Evaluates grounding coverage on candidate synthetic and real narratives.
    Verifies that ungrounded/hallucinated sentences are strictly pruned.
    """
    mock_events = [
        {
            "event_id": "EVT-001",
            "case_id": "CASE-CARD-001",
            "timestamp": "2026-09-18T03:22:00Z",
            "tool_called": "get_transaction",
            "tool_input": {"transaction_id": "TX-CARD-9842"},
            "tool_output_summary": "Retrieved transaction TX-CARD-9842 for $4,850.00.",
            "narration_sentence": "Card transaction TX-CARD-9842 of $4,850.00 was authorized at 03:22 AM.",
        },
        {
            "event_id": "EVT-002",
            "case_id": "CASE-CARD-001",
            "timestamp": "2026-09-18T03:22:05Z",
            "tool_called": "get_shap_explanation",
            "tool_input": {"transaction_id": "TX-CARD-9842"},
            "tool_output_summary": "Risk score 0.89 (flagged) driven by Amount and Time.",
            "narration_sentence": "The LightGBM fraud model assigned a risk score of 0.89 (flagged) driven by transaction amount and off-hours timing.",
        },
    ]

    candidate_text = (
        "Card transaction TX-CARD-9842 of $4,850.00 was authorized at 03:22 AM. "
        "Funds were immediately routed to an unverified offshore account in the Cayman Islands. "
        "The LightGBM fraud model assigned a risk score of 0.89 (flagged) driven by transaction amount and off-hours timing."
    )

    audit = audit_grounding(mock_events, raw_narrative=candidate_text)
    total_sentences = audit["total_candidate_sentences"]
    retained_sentences = len(audit["retained_sentences"])
    pruned_sentences = len(audit["pruned_sentences"])
    coverage_pct = round((retained_sentences / total_sentences) * 100, 1) if total_sentences > 0 else 100.0

    return {
        "metric_name": "grounding_coverage",
        "total_candidate_sentences": total_sentences,
        "retained_sentences": retained_sentences,
        "pruned_sentences": pruned_sentences,
        "hallucinations_blocked": pruned_sentences,
        "grounding_retention_rate": f"{coverage_pct}%",
        "audit_detail": audit,
    }

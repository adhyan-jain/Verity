"""
Unit tests for the Narrative-Model Consistency evaluation harness.
"""
import pytest
from agent.eval_consistency import (
    check_verdict_consistency,
    check_feature_overlap,
    evaluate_grounding_coverage,
    evaluate_narrative_model_consistency,
    DEMO_FRAUD_PERTURBATIONS
)
from agent.llm import VerityLLMClient


def test_check_verdict_consistency():
    # Case 1: Expected flagged
    assert check_verdict_consistency("Transaction is flagged due to high velocity.", "flagged") is True
    assert check_verdict_consistency("The transaction is suspicious and anomalous.", "flagged") is True
    assert check_verdict_consistency("Transaction is clear and safe.", "flagged") is False

    # Case 2: Expected clear
    assert check_verdict_consistency("Transaction is cleared of fraud risk.", "clear") is True
    assert check_verdict_consistency("Risk was reduced and amount is normal.", "clear") is True
    assert check_verdict_consistency("Transaction is fraudulent and flagged.", "clear") is False


def test_check_feature_overlap():
    top_shap = ["V14", "Amount", "V4"]

    # Matching narrative mentions V14 or Amount
    narrative_1 = "The model flags this primarily because of anomalous V14 behavior."
    assert check_feature_overlap(narrative_1, top_shap[:2]) is True

    narrative_2 = "High transaction amount of $5,000 triggered the alarm."
    assert check_feature_overlap(narrative_2, top_shap[:2]) is True

    # Non-matching narrative
    narrative_3 = "The user logged in from an unknown device and browser."
    assert check_feature_overlap(narrative_3, ["V14", "V4"]) is False


def test_demo_fraud_perturbations_structure():
    assert len(DEMO_FRAUD_PERTURBATIONS) >= 3
    total_perturbations = sum(len(item["perturbations"]) for item in DEMO_FRAUD_PERTURBATIONS)
    assert total_perturbations >= 9  # At least 3 per case


def test_eval_consistency_deterministic_mode():
    # When client has no api key (deterministic mode), evaluate_narrative_model_consistency
    # must return is_deterministic_mode=True and the explicit caveat display
    mock_client = VerityLLMClient()
    mock_client.api_key = None
    mock_client.provider = "builtin"
    res = evaluate_narrative_model_consistency(client=mock_client)
    assert res["is_deterministic_mode"] is True
    assert res["consistency_display"] == "N/A — deterministic mode, result is not meaningful"


def test_eval_grounding_coverage():
    res = evaluate_grounding_coverage()
    assert res["metric_name"] == "grounding_coverage"
    assert res["total_candidate_sentences"] == 3
    assert res["retained_sentences"] == 2
    assert res["pruned_sentences"] == 1
    assert res["hallucinations_blocked"] == 1

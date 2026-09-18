"""
Unit tests for Grounding Enforcement Filter (agent/grounding.py).
Tests code-level hallucination purging and citation integrity.
"""

import pytest
from agent.grounding import ground_narrative, audit_grounding, _split_into_sentences


def test_split_into_sentences_preserves_currency_and_decimals():
    text = "The transaction was $4,850.00 at 03:22 AM. Second sentence begins here. Ratio was 3.14."
    sentences = _split_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "The transaction was $4,850.00 at 03:22 AM."
    assert sentences[1] == "Second sentence begins here."
    assert sentences[2] == "Ratio was 3.14."


def test_deterministic_assembly():
    trace_events = [
        {
            "event_id": "EVT-101",
            "narration_sentence": "Retrieved transaction TX-CARD-9842 for $4,850.00 at 03:22 AM.",
            "tool_output_summary": "TX-CARD-9842 retrieved."
        },
        {
            "event_id": "EVT-102",
            "narration_sentence": "SHAP feature attribution indicates elevated risk driven by amount (+0.42).",
            "tool_output_summary": "SHAP risk score 0.89."
        }
    ]

    narrative, verified_events = ground_narrative(trace_events)
    assert len(verified_events) == 2
    assert "Retrieved transaction TX-CARD-9842 for $4,850.00 at 03:22 AM." in narrative
    assert "SHAP feature attribution indicates elevated risk" in narrative


def test_adversarial_hallucination_stripping():
    trace_events = [
        {
            "event_id": "EVT-101",
            "narration_sentence": "Retrieved transaction TX-CARD-9842 for $4,850.00 at 03:22 AM.",
            "tool_output_summary": "TX-CARD-9842 retrieved.",
            "tool_input": {"transaction_id": "TX-CARD-9842"}
        }
    ]

    # Deliberately inject hallucinated unverified claims
    hallucinated_text = (
        "Retrieved transaction TX-CARD-9842 for $4,850.00 at 03:22 AM. "
        "The subject fled to an offshore bank in the Cayman Islands to launder cartel funds."
    )

    audit = audit_grounding(trace_events, hallucinated_text)
    assert audit["total_candidate_sentences"] == 2
    assert audit["retained_count"] == 1
    assert audit["pruned_count"] == 1
    assert "Cayman Islands" not in audit["grounded_narrative"]
    assert "Retrieved transaction TX-CARD-9842" in audit["grounded_narrative"]


def test_empty_raw_narrative_uses_trace_events():
    trace_events = [
        {
            "event_id": "EVT-301",
            "narration_sentence": "Traversed synthetic network finding circular fund flow.",
            "tool_output_summary": "Round tripping cycle."
        }
    ]

    narrative, _ = ground_narrative(trace_events, raw_narrative="")
    assert narrative == "Traversed synthetic network finding circular fund flow."

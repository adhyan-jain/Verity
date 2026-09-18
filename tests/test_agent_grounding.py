"""
Unit tests for Strict Evidence-Only Grounding Filter (agent/grounding.py).
Tests code-level hallucination purging, rejection of unsupported speculative claims,
and preservation of strictly grounded factual statements.
"""

from agent.grounding import (
    _split_into_sentences,
    audit_grounding,
    ground_narrative,
    is_sentence_strictly_grounded,
)


def test_split_into_sentences_preserves_currency_and_decimals():
    text = "The transaction was $4,850.00 at 03:22 AM. Second sentence begins here. Ratio was 3.14."
    sentences = _split_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "The transaction was $4,850.00 at 03:22 AM."
    assert sentences[1] == "Second sentence begins here."
    assert sentences[2] == "Ratio was 3.14."


def test_critique_offshore_hallucination_rejected():
    """
    Specifically tests the critique's exact failure mode:
    Evidence: "Transaction TX-100 had amount 4850 and occurred at 03:22."
    Candidate: "Transaction TX-100 had amount 4850 and was transferred to an offshore account."
    Must be strictly rejected despite high lexical/token overlap!
    """
    trace_events = [
        {
            "event_id": "EVT-101",
            "narration_sentence": "Transaction TX-100 had amount 4850 and occurred at 03:22.",
            "tool_output_summary": "TX-100 4850",
            "tool_input": {"transaction_id": "TX-100"},
        }
    ]

    candidate = (
        "Transaction TX-100 had amount 4850 and was transferred to an offshore account."
    )
    is_grounded, reason = is_sentence_strictly_grounded(
        candidate, [e["narration_sentence"] for e in trace_events], trace_events
    )

    assert is_grounded is False
    assert "offshore" in reason.lower()


def test_unsupported_entity_id_rejected():
    trace_events = [
        {
            "event_id": "EVT-201",
            "narration_sentence": "Account history analysis for ACC-1092 detected a velocity spike.",
            "tool_output_summary": "ACC-1092 4 transactions",
            "tool_input": {"account_id": "ACC-1092"},
        }
    ]

    # ACC-9999 does not exist in trace events
    candidate = "Funds were subsequently transferred to ACC-9999 without authorization."
    is_grounded, reason = is_sentence_strictly_grounded(
        candidate, [e["narration_sentence"] for e in trace_events], trace_events
    )

    assert is_grounded is False
    assert "acc-9999" in reason.lower()


def test_unsupported_numeric_amount_rejected():
    trace_events = [
        {
            "event_id": "EVT-101",
            "narration_sentence": "Retrieved transaction TX-CARD-9842 for $4,850.00 at 03:22 AM.",
            "tool_output_summary": "TX-CARD-9842 amount 4850.00",
            "tool_input": {"transaction_id": "TX-CARD-9842"},
        }
    ]

    # $99,000.00 is an unbacked fabricated figure
    candidate = "Retrieved transaction TX-CARD-9842 for $99,000.00 at 03:22 AM."
    is_grounded, reason = is_sentence_strictly_grounded(
        candidate, [e["narration_sentence"] for e in trace_events], trace_events
    )

    assert is_grounded is False
    assert "numeric" in reason.lower() or "99,000" in reason


def test_deterministic_assembly_and_clean_audit():
    trace_events = [
        {
            "event_id": "EVT-101",
            "narration_sentence": "Retrieved transaction TX-CARD-9842 for $4,850.00 at 03:22 AM.",
            "tool_output_summary": "TX-CARD-9842 retrieved.",
        },
        {
            "event_id": "EVT-102",
            "narration_sentence": "SHAP feature attribution indicates elevated risk driven by Amount (+0.42).",
            "tool_output_summary": "SHAP risk score 0.89.",
        },
    ]

    narrative, verified_events = ground_narrative(trace_events)
    assert len(verified_events) == 2
    assert "Retrieved transaction TX-CARD-9842 for $4,850.00 at 03:22 AM." in narrative
    assert "SHAP feature attribution indicates elevated risk" in narrative


def test_audit_grounding_separates_retained_and_pruned():
    trace_events = [
        {
            "event_id": "EVT-101",
            "narration_sentence": "Retrieved transaction TX-CARD-9842 for $4,850.00 at 03:22 AM.",
            "tool_output_summary": "TX-CARD-9842 retrieved.",
            "tool_input": {"transaction_id": "TX-CARD-9842"},
        }
    ]

    raw_llm_output = (
        "Retrieved transaction TX-CARD-9842 for $4,850.00 at 03:22 AM. "
        "The customer transferred the proceeds to a Cayman Islands tax haven."
    )

    audit = audit_grounding(trace_events, raw_llm_output)
    assert audit["total_candidate_sentences"] == 2
    assert audit["retained_count"] == 1
    assert audit["pruned_count"] == 1
    assert "Cayman" not in audit["grounded_narrative"]
    assert "TX-CARD-9842" in audit["grounded_narrative"]

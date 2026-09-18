"""
Grounding Enforcement Filter.
Person C: Strict Evidence-Only Grounding.
Every narrative sentence must be strictly backed by an actual AgentTraceEvent.
Eliminates any hallucinated claims, entities, or offshore assertions at code level.
"""

import re
from typing import Any

# Suspicious / speculative terminology that cannot appear without explicit evidence
SPECULATIVE_PREDICATES = {
    "offshore",
    "cayman",
    "swiss",
    "panama",
    "cyprus",
    "shell",
    "haven",
    "cartel",
    "smuggling",
    "bribe",
    "extortion",
    "terrorist",
    "laundering ring",
    "mule",
    "straw man",
    "front company",
    "unregistered transmitter",
    "unauthorized access",
    "hacked",
    "stolen credentials",
}


def _split_into_sentences(text: str) -> list[str]:
    """
    Splits text into sentences while protecting decimals ($4,850.00),
    timestamps (03:22 AM), and common technical tokens.
    """
    if not text:
        return []

    # Protect decimals in currency: $4,850.00 -> $4,850<DOT>00
    cleaned = re.sub(r"(\$\d[\d,]*)\.(\d+)", r"\1<DOT>\2", text)
    cleaned = re.sub(r"(\b\d+)\.(\d+)\b", r"\1<DOT>\2", cleaned)
    # Protect common abbreviations
    cleaned = re.sub(
        r"\b(e\.g\.|i\.e\.|vs\.|approx\.|dr\.|mr\.)",
        lambda m: m.group(1).replace(".", "<DOT>"),
        cleaned,
        flags=re.IGNORECASE,
    )
    # Protect AM/PM
    cleaned = re.sub(
        r"\b(A\.M\.|P\.M\.)",
        lambda m: m.group(1).replace(".", "<DOT>"),
        cleaned,
        flags=re.IGNORECASE,
    )

    # Split on sentence terminals followed by whitespace
    raw_sentences = re.split(r"(?<=[.!?])\s+", cleaned)

    sentences = []
    for s in raw_sentences:
        restored = s.replace("<DOT>", ".").strip()
        if restored:
            sentences.append(restored)

    return sentences


def _extract_tokens(text: str) -> set[str]:
    """Extracts alphanumeric tokens in lowercase."""
    return set(re.findall(r"[a-zA-Z0-9_\-]+", text.lower()))


def _extract_factual_entities(text: str) -> dict[str, set[str]]:
    """
    Extracts high-risk factual assertions:
    - identifiers: TX-..., ACC-..., EVT-...
    - numbers and monetary values: amounts, decimals
    - times: HH:MM, timestamps
    - content words: nouns/adjectives/predicates
    """
    clean = text.lower()
    ids = set(re.findall(r"\b(?:tx|acc|evt|case|flag)-[a-zA-Z0-9_\-]+\b", clean))
    numbers = set(re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?\b", text))
    times = set(re.findall(r"\b\d{1,2}:\d{2}(?:\s*[ap]m)?\b", clean))
    words = _extract_tokens(clean)

    return {"ids": ids, "numbers": numbers, "times": times, "words": words}


def find_supporting_event_id(
    sentence: str, trace_events: list[dict[str, Any]]
) -> str | None:
    """Finds the event_id in trace_events that directly supports the given sentence."""
    s_norm = sentence.strip().lower().rstrip(".!?;:")
    # 1. Exact match with an event narration sentence
    for evt in trace_events:
        evt_sent = evt.get("narration_sentence", "").strip().lower().rstrip(".!?;:")
        if s_norm == evt_sent:
            return evt.get("event_id")

    # 2. Token overlap match
    s_tokens = _extract_tokens(s_norm)
    best_id = None
    best_overlap = 0
    for evt in trace_events:
        evt_text = (
            evt.get("narration_sentence", "")
            + " "
            + evt.get("tool_output_summary", "")
            + " "
            + str(evt.get("tool_input", {}))
        ).lower()
        evt_tokens = _extract_tokens(evt_text)
        overlap = len(s_tokens & evt_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_id = evt.get("event_id")

    return best_id or (trace_events[0].get("event_id") if trace_events else None)


def is_sentence_strictly_grounded(
    candidate: str, approved_sentences: list[str], trace_events: list[dict[str, Any]]
) -> tuple[bool, str | None]:
    """
    Strict evidence-only grounding rule:
    1. Exact normalized match with an approved narration_sentence -> PASS
    2. Factual containment check:
       - Every entity ID (TX-..., ACC-..., EVT-...) must be in trace evidence.
       - Every numeric figure / currency amount must be strictly verified against trace evidence.
       - NO unbacked speculative terms (offshore, shell, cartel, etc.) unless explicitly present in trace evidence.
       - Core informative terms must be supported by the evidence corpus.
    """
    cand_norm = candidate.strip().lower().rstrip(".!?;:")

    # Rule 1: Exact normalized match with an approved narration sentence.
    # (A loose substring check here previously let a short, unrelated
    # sentence short-circuit as grounded whenever it happened to be a
    # substring of an approved sentence or vice versa, bypassing every
    # stricter fact check below.)
    for app in approved_sentences:
        app_norm = app.strip().lower().rstrip(".!?;:")
        if cand_norm == app_norm:
            return True, None

    # Build the complete verified evidence corpus from all trace events
    evidence_text_parts = []
    for evt in trace_events:
        evidence_text_parts.append(evt.get("narration_sentence", ""))
        evidence_text_parts.append(evt.get("tool_output_summary", ""))
        evidence_text_parts.append(str(evt.get("tool_input", {})))
        evidence_text_parts.append(evt.get("event_id", ""))

    evidence_corpus = " ".join(evidence_text_parts)
    evidence_facts = _extract_factual_entities(evidence_corpus)
    cand_facts = _extract_factual_entities(candidate)

    # Check 1: Unsupported entity IDs
    for ident in cand_facts["ids"]:
        if (
            not any(ident in ev_id for ev_id in evidence_facts["ids"])
            and ident not in evidence_corpus.lower()
        ):
            return False, f"Unsupported entity ID: '{ident}'"

    # Check 2: Unsupported numbers or amounts (normalized for commas/decimals)
    evidence_corpus_clean = evidence_corpus.replace(",", "")
    for num in cand_facts["numbers"]:
        num_clean = num.replace(",", "")
        if (
            len(num_clean) > 1
            and num_clean not in evidence_corpus_clean
            and num not in evidence_corpus
        ):
            return False, f"Unsupported numeric value: '{num}'"

    # Check 3: Unsupported speculative terms (The Offshore Account Test)
    cand_words = cand_facts["words"]
    for spec in SPECULATIVE_PREDICATES:
        if spec in cand_words:
            # Check if this speculative word exists in the evidence corpus
            if spec not in evidence_facts["words"]:
                return False, f"Unsupported speculative claim: '{spec}'"

    # Check 4: Substantial factual containment (reject sentences introducing new facts)
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "was",
        "were",
        "and",
        "or",
        "to",
        "for",
        "in",
        "on",
        "at",
        "of",
        "by",
        "this",
        "that",
        "it",
        "with",
        "from",
        "has",
        "have",
        "had",
        "been",
        "indicates",
        "detected",
        "retrieved",
        "analysis",
        "found",
        "exceeded",
        "representing",
        "processed",
        "also",
        "then",
        "furthermore",
        "which",
        "as",
        "into",
        "within",
        "exhibits",
    }
    informative_cand_words = cand_words - stopwords

    if not informative_cand_words:
        return False, "Sentence contains no verifiable informative content"

    # Check how many informative words are completely absent from evidence
    unsupported_words = informative_cand_words - evidence_facts["words"]
    # Allow at most 1 connecting word variation; all key domain terms must be grounded
    if len(unsupported_words) > 1:
        return False, f"Unsupported factual terms: {unsupported_words}"

    return True, None


def ground_narrative(
    trace_events: list[dict[str, Any]], raw_narrative: str | None = None
) -> tuple[str, list[dict[str, Any]]]:
    """
    Grounding rule: The agent's narrative field can ONLY be assembled from
    sentences that pass strict evidence verification against AgentTraceEvents.

    If raw_narrative (from LLM) is passed:
    - Splits into candidate sentences.
    - Runs strict evidence verification.
    - Strips any sentence with unsupported claims.
    - If all candidate sentences fail, falls back to approved trace narration sentences.
    """
    valid_events = [
        e for e in trace_events if e.get("event_id") and e.get("narration_sentence")
    ]
    approved_sentences = [e["narration_sentence"].strip() for e in valid_events]

    if not raw_narrative or not raw_narrative.strip():
        grounded_narrative = " ".join(approved_sentences)
        return grounded_narrative, valid_events

    candidate_sentences = _split_into_sentences(raw_narrative)
    grounded_sentences = []

    for candidate in candidate_sentences:
        is_grounded, _ = is_sentence_strictly_grounded(
            candidate, approved_sentences, valid_events
        )
        if is_grounded:
            grounded_sentences.append(candidate)

    # Fallback to approved sentences if all candidate sentences were stripped
    if not grounded_sentences:
        grounded_narrative = " ".join(approved_sentences)
    else:
        grounded_narrative = " ".join(grounded_sentences)

    return grounded_narrative, valid_events


def audit_grounding(
    trace_events: list[dict[str, Any]], raw_narrative: str
) -> dict[str, Any]:
    """
    Audit tool for verifying grounding filter behavior and tracking dropped sentences.
    """
    valid_events = [
        e for e in trace_events if e.get("event_id") and e.get("narration_sentence")
    ]
    approved_sentences = [e["narration_sentence"].strip() for e in valid_events]
    candidate_sentences = _split_into_sentences(raw_narrative)

    retained = []
    pruned = []
    rejection_reasons = {}
    supporting_event_ids = {}

    for candidate in candidate_sentences:
        is_grounded, reason = is_sentence_strictly_grounded(
            candidate, approved_sentences, valid_events
        )
        if is_grounded:
            retained.append(candidate)
            supp_id = find_supporting_event_id(candidate, valid_events)
            if supp_id:
                supporting_event_ids[candidate] = supp_id
        else:
            pruned.append(candidate)
            rejection_reasons[candidate] = reason or "Failed evidence verification"

    return {
        "total_candidate_sentences": len(candidate_sentences),
        "retained_count": len(retained),
        "pruned_count": len(pruned),
        "retained_sentences": retained,
        "pruned_sentences": pruned,
        "rejection_reasons": rejection_reasons,
        "supporting_event_ids": supporting_event_ids,
        "grounded_narrative": " ".join(retained)
        if retained
        else " ".join(approved_sentences),
    }

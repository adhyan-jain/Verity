"""
Grounding Enforcement Filter.
Person C: Enforces that narrative sentences must cite a valid AgentTraceEvent ID.
Filters out any hallucinated sentences at code level before output reaches the dashboard.
"""

import re
from typing import List, Dict, Any, Tuple, Optional


def _split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences while protecting decimals ($4,850.00),
    timestamps (03:22 AM), and common technical tokens.
    """
    if not text:
        return []

    # Replace periods in common currency/decimal formats with a placeholder
    # e.g., $4,850.00 -> $4,850<DOT>00
    cleaned = re.sub(r'(\$\d[\d,]*)\.(\d+)', r'\1<DOT>\2', text)
    cleaned = re.sub(r'(\b\d+)\.(\d+)\b', r'\1<DOT>\2', cleaned)
    # Protect common abbreviations
    cleaned = re.sub(r'\b(e\.g\.|i\.e\.|vs\.|approx\.|dr\.|mr\.)', lambda m: m.group(1).replace('.', '<DOT>'), cleaned, flags=re.IGNORECASE)
    # Protect AM/PM periods
    cleaned = re.sub(r'\b(A\.M\.|P\.M\.)', lambda m: m.group(1).replace('.', '<DOT>'), cleaned, flags=re.IGNORECASE)

    # Split on sentence terminals (. ! ?) followed by whitespace or end of string
    raw_sentences = re.split(r'(?<=[.!?])\s+', cleaned)

    sentences = []
    for s in raw_sentences:
        restored = s.replace('<DOT>', '.').strip()
        if restored:
            sentences.append(restored)

    return sentences


def _normalize_tokens(text: str) -> set:
    """Extracts lowercase alpha-numeric tokens for similarity comparison."""
    return set(re.findall(r'[a-zA-Z0-9_\-]+', text.lower()))


def is_sentence_grounded(candidate: str, approved_sentences: List[str], trace_events: List[Dict[str, Any]]) -> bool:
    """
    Determines if a candidate sentence is strictly grounded in the approved trace events:
    1. Direct match with an approved narration_sentence.
    2. Explicit citation of an approved event_id (e.g. [EVT-101] or EVT-101).
    3. Token overlap threshold (>70% of candidate informative tokens present in approved events).
    """
    cand_norm = candidate.strip().lower()

    # 1. Direct or substring matching against approved narration sentences
    for approved in approved_sentences:
        app_norm = approved.strip().lower()
        if cand_norm in app_norm or app_norm in cand_norm:
            return True

    # 2. Check for explicit event_id citations
    for event in trace_events:
        evt_id = event.get("event_id", "").lower()
        if evt_id and evt_id in cand_norm:
            return True

    # 3. Informative token containment check
    candidate_tokens = _normalize_tokens(candidate)
    # Remove common stop words
    stopwords = {"the", "a", "an", "is", "was", "were", "and", "or", "to", "for", "in", "on", "at", "of", "by", "this", "that", "it"}
    informative_tokens = candidate_tokens - stopwords

    if not informative_tokens:
        return False

    approved_corpus = set()
    for approved in approved_sentences:
        approved_corpus.update(_normalize_tokens(approved))
    for event in trace_events:
        approved_corpus.update(_normalize_tokens(event.get("tool_output_summary", "")))
        approved_corpus.update(_normalize_tokens(str(event.get("tool_input", {}))))

    overlap = informative_tokens.intersection(approved_corpus)
    overlap_ratio = len(overlap) / len(informative_tokens)

    # Must have high overlap with facts emitted by the trace events
    return overlap_ratio >= 0.70


def ground_narrative(
    trace_events: List[Dict[str, Any]], 
    raw_narrative: Optional[str] = None
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Grounding rule: The agent's narrative field can ONLY be assembled from
    narration_sentence values that came from verified AgentTraceEvents.

    If a raw_narrative (from an LLM call) is passed:
    - Splits it into candidate sentences.
    - Strips any candidate sentence that is not grounded in the emitted AgentTraceEvents.
    - If all candidate sentences are dropped, falls back to trace_events narration_sentences.
    """
    # 1. Collect verified narration sentences from valid trace events
    valid_events = [e for e in trace_events if e.get("event_id") and e.get("narration_sentence")]
    approved_sentences = [e["narration_sentence"].strip() for e in valid_events]

    # 2. If no raw narrative was provided, perform deterministic assembly
    if not raw_narrative or not raw_narrative.strip():
        grounded_narrative = " ".join(approved_sentences)
        return grounded_narrative, valid_events

    # 3. If raw narrative was provided, filter it sentence-by-sentence
    candidate_sentences = _split_into_sentences(raw_narrative)
    grounded_sentences = []

    for candidate in candidate_sentences:
        if is_sentence_grounded(candidate, approved_sentences, valid_events):
            grounded_sentences.append(candidate)

    # 4. Fallback if the LLM hallucinated entirely and everything was stripped
    if not grounded_sentences:
        grounded_narrative = " ".join(approved_sentences)
    else:
        grounded_narrative = " ".join(grounded_sentences)

    return grounded_narrative, valid_events


def audit_grounding(trace_events: List[Dict[str, Any]], raw_narrative: str) -> Dict[str, Any]:
    """
    Audit tool for verifying grounding filter behavior and tracking dropped sentences.
    """
    valid_events = [e for e in trace_events if e.get("event_id") and e.get("narration_sentence")]
    approved_sentences = [e["narration_sentence"].strip() for e in valid_events]
    candidate_sentences = _split_into_sentences(raw_narrative)

    retained = []
    pruned = []

    for candidate in candidate_sentences:
        if is_sentence_grounded(candidate, approved_sentences, valid_events):
            retained.append(candidate)
        else:
            pruned.append(candidate)

    return {
        "total_candidate_sentences": len(candidate_sentences),
        "retained_count": len(retained),
        "pruned_count": len(pruned),
        "retained_sentences": retained,
        "pruned_sentences": pruned,
        "grounded_narrative": " ".join(retained) if retained else " ".join(approved_sentences)
    }

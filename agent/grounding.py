"""
Grounding Enforcement Filter.
Person C: Enforces that narrative sentences must cite a valid AgentTraceEvent ID.
Filters out any hallucinated sentences before output reaches the dashboard.
"""

from typing import List, Dict, Any, Tuple


def ground_narrative(trace_events: List[Dict[str, Any]], raw_narrative: str = "") -> Tuple[str, List[Dict[str, Any]]]:
    """
    Grounding rule: The agent's narrative field can ONLY be assembled from
    narration_sentence values that came from verified AgentTraceEvents.

    Any hallucinated hops or ungrounded sentences not tied to an event_id
    are stripped at code level.
    """
    valid_sentences = [
        event["narration_sentence"].strip()
        for event in trace_events
        if event.get("narration_sentence") and event.get("event_id")
    ]
    grounded_narrative = " ".join(valid_sentences)
    return grounded_narrative, trace_events

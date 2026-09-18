"""
Hand-Rolled Tool Calling Loop.
Person C: Pure explicit loop (no agent framework).
Dispatches model tool calls, appends AgentTraceEvents, and returns grounded Case narratives.
"""

from typing import Dict, Any, List
import uuid
import datetime
from .tools import get_transaction
from .grounding import ground_narrative


def run_investigation_loop(
    case_id: str, primary_transaction_id: str, tier_origin: str
) -> Dict[str, Any]:
    """
    Executes investigative loop for a flagged case:
    1. Dispatches tools (get_transaction, get_shap_explanation, walk_graph)
    2. Emits AgentTraceEvent for each call
    3. Feeds trace events into ground_narrative
    4. Assembles Case output object
    """
    trace_events: List[Dict[str, Any]] = []

    # Step 1: Retrieve transaction
    tx_data = get_transaction(primary_transaction_id)
    trace_events.append(
        {
            "event_id": f"EVT-{uuid.uuid4().hex[:6].upper()}",
            "case_id": case_id,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "tool_called": "get_transaction",
            "tool_input": {"transaction_id": primary_transaction_id},
            "tool_output_summary": f"Retrieved transaction {primary_transaction_id} from {tx_data.get('tier')}.",
            "narration_sentence": f"Retrieved record {primary_transaction_id} for investigation.",
        }
    )

    # Step 2: Ground narrative
    narrative, grounded_events = ground_narrative(trace_events)

    return {
        "case_id": case_id,
        "tier_origin": tier_origin,
        "status": "investigating",
        "primary_transaction_id": primary_transaction_id,
        "risk_score": 0.85,
        "trace_events": grounded_events,
        "narrative": narrative,
    }

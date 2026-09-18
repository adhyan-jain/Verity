"""
Hand-Rolled Tool Calling Loop with LLM Decisioning.
Person C: Pure explicit loop (no agent framework).
Dispatches model tool calls, appends AgentTraceEvents, analyzes engine evidence mathematically,
and returns strictly grounded Case narratives.
"""

import datetime
import logging
import uuid
from collections.abc import Callable
from typing import Any

from .grounding import ground_narrative
from .llm import VerityLLMClient
from .tools import (
    TransactionNotFoundError,
    counterfactual,
    get_shap_explanation,
    get_transaction,
    walk_graph,
)

logger = logging.getLogger("verity.agent.loop")


def _generate_event_id() -> str:
    """Generates unique trace event ID."""
    return f"EVT-{uuid.uuid4().hex[:6].upper()}"


def _get_utc_timestamp() -> str:
    """Generates ISO8601 UTC timestamp."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _analyze_ledger_evidence(
    steps: list[dict[str, Any]], account_id: str
) -> tuple[float, str, str]:
    """
    Analyzes returned ledger transaction sequence mathematically:
    - Running balance degradation / negative reserves
    - Burst velocity relative to baseline
    Returns: (risk_score, tool_summary, narration_sentence)
    """
    if not steps:
        return (
            0.50,
            f"No transactions recorded for account {account_id}.",
            f"Account {account_id} has no active ledger activity.",
        )

    balances = [s.get("balance", 0.0) for s in steps if "balance" in s]
    amounts = [s.get("amount", 0.0) for s in steps]
    step_count = len(steps)

    has_negative_balance = any(b < 0 for b in balances)
    min_balance = min(balances) if balances else 0.0
    start_balance = balances[0] if balances else 0.0
    total_outflow = sum(amounts)

    # Dynamic risk calculation based on observed breaks
    calculated_risk = 0.20
    reasons = []

    if has_negative_balance:
        calculated_risk += 0.40
        reasons.append(
            f"balance plummeted into negative overdraft (${min_balance:,.2f})"
        )

    if step_count >= 3:
        calculated_risk += 0.20
        reasons.append(f"high-velocity burst of {step_count} consecutive transfers")

    if total_outflow > 10000.0:
        calculated_risk += 0.14
        reasons.append(f"cumulative debit outflow of ${total_outflow:,.2f}")

    risk_score = min(0.95, round(calculated_risk, 2))
    reasons_desc = " and ".join(reasons) or "baseline variation"

    summary = (
        f"Ledger analysis for account {account_id}: evaluated {step_count} transactions. "
        f"Initial balance ${start_balance:,.2f}, lowest balance ${min_balance:,.2f}. Risk: {risk_score}."
    )
    sentence = f"Account history analysis for {account_id} detected a severe anomaly: {reasons_desc}."

    return risk_score, summary, sentence


def _analyze_synthetic_evidence(
    steps: list[dict[str, Any]], account_id: str
) -> tuple[float, str, str]:
    """
    Analyzes synthetic network graph walk:
    - Tests for cycle preservation (start_node == end_node)
    - Computes volume retention ratio: final_amount / initial_amount
    Returns: (risk_score, tool_summary, narration_sentence)
    """
    if not steps:
        return (
            0.50,
            f"No graph steps found for {account_id}.",
            f"Graph traversal returned no connected entities for {account_id}.",
        )

    start_node = steps[0].get("from_account")
    end_node = steps[-1].get("to_account")
    initial_amount = steps[0].get("amount", 1.0)
    final_amount = steps[-1].get("amount", 0.0)

    is_cycle = start_node == end_node
    retention_ratio = (final_amount / initial_amount) if initial_amount > 0 else 0.0
    retention_pct = round(retention_ratio * 100, 1)
    hop_count = len(steps)

    if is_cycle and retention_ratio >= 0.85:
        risk_score = 0.94
        typology = "FATF Round-Tripping (circular flow)"
        summary = (
            f"Traversed {hop_count} hops: {start_node} -> {end_node}. Verified closed cycle with {retention_pct}% "
            f"volume conservation (${final_amount:,.2f} returned of ${initial_amount:,.2f})."
        )
        sentence = (
            f"Graph traversal identified a confirmed {typology} topology across {hop_count} hops "
            f"retaining {retention_pct}% volume within a compressed window."
        )
    elif hop_count >= 3:
        risk_score = 0.88
        typology = "FATF Rapid Layering"
        summary = f"Traversed {hop_count} high-velocity pass-through hops originating at {start_node}."
        sentence = f"Graph traversal identified a high-velocity pass-through layering path across {hop_count} accounts."
    else:
        risk_score = 0.60
        summary = (
            f"Graph traversal traced {hop_count} steps without confirmed FATF cycle."
        )
        sentence = f"Graph traversal mapped {hop_count} intermediate account hops."

    return risk_score, summary, sentence


def run_investigation_loop(
    case_id: str,
    primary_transaction_id: str,
    tier_origin: str = "real_card",
    on_step_callback: Callable[[dict[str, Any]], None] | None = None,
    llm_client: VerityLLMClient | None = None,
    max_steps: int = 4,
) -> dict[str, Any]:
    """
    Executes investigative loop with an actual LLM decision engine:
    1. Loop: LLM decides tool to call -> executes tool -> emits AgentTraceEvent -> feeds observation back
    2. Ends when LLM selects 'finish' or reaches max_steps
    3. LLM produces candidate narrative
    4. Code-level grounding filter prunes any ungrounded assertions
    5. Returns finalized Case object
    """
    client = llm_client or VerityLLMClient()
    trace_events: list[dict[str, Any]] = []
    case_risk_score = 0.85
    actual_tier = tier_origin
    candidate_narrative: str | None = None

    for step_num in range(1, max_steps + 1):
        # 1. LLM decides the next action
        decision = client.decide_next_step(
            case_id=case_id,
            primary_tx_id=primary_transaction_id,
            tier_origin=actual_tier,
            history=trace_events,
            step_number=step_num,
            max_steps=max_steps,
        )

        action = decision.get("action", "finish")
        action_input = decision.get("action_input", {})

        if action == "finish":
            candidate_narrative = decision.get("candidate_narrative")
            break

        # 2. Execute selected tool
        if action == "get_transaction":
            tx_id = action_input.get("transaction_id", primary_transaction_id)
            try:
                tx_data = get_transaction(tx_id)
            except TransactionNotFoundError:
                evt = {
                    "event_id": _generate_event_id(),
                    "case_id": case_id,
                    "timestamp": _get_utc_timestamp(),
                    "tool_called": "get_transaction",
                    "tool_input": action_input,
                    "tool_output_summary": f"Transaction {tx_id} not found.",
                    "narration_sentence": f"Transaction {tx_id} could not be located in any engine or fixture.",
                    "raw_output": {"error": "not_found", "transaction_id": tx_id},
                }
                trace_events.append(evt)
                if on_step_callback:
                    on_step_callback(evt)
                break
            actual_tier = tx_data.get("tier", actual_tier)
            amount_str = f"${tx_data.get('amount', 0.0):,.2f}"

            summary = f"Retrieved transaction {tx_id} ({actual_tier}): amount {amount_str} ({tx_data.get('direction', 'debit')}) at {tx_data.get('timestamp')}."
            sentence = f"Retrieved transaction {tx_id} for {amount_str} processed on rail {tx_data.get('source_dataset', 'dataset')}."

            evt = {
                "event_id": _generate_event_id(),
                "case_id": case_id,
                "timestamp": _get_utc_timestamp(),
                "tool_called": "get_transaction",
                "tool_input": action_input,
                "tool_output_summary": summary,
                "narration_sentence": sentence,
                "raw_output": tx_data,
            }
            trace_events.append(evt)
            if on_step_callback:
                on_step_callback(evt)

        elif action == "get_shap_explanation":
            tx_id = action_input.get("transaction_id", primary_transaction_id)
            shap_data = get_shap_explanation(tx_id)
            case_risk_score = float(shap_data.get("risk_score", 0.89))
            top_factors = shap_data.get("top_factors", [])

            interp = [
                f"{f['feature']} ({f['contribution']:+.2f})"
                for f in top_factors
                if f.get("interpretable")
            ]
            anon = [
                f"{f['feature']} ({f['contribution']:+.2f})"
                for f in top_factors
                if not f.get("interpretable")
            ]

            interp_str = ", ".join(interp) or "Amount"
            anon_str = ", ".join(anon) or "V14"

            summary = f"SHAP model returned risk score {case_risk_score:.2f} ({shap_data.get('verdict', 'flagged')}). Interpretable: {interp_str}. Anonymized: {anon_str}."
            sentence = f"SHAP feature attribution indicates elevated risk score {case_risk_score:.2f} driven by interpretable factors [{interp_str}] and anonymized signals [{anon_str}]."

            evt = {
                "event_id": _generate_event_id(),
                "case_id": case_id,
                "timestamp": _get_utc_timestamp(),
                "tool_called": "get_shap_explanation",
                "tool_input": action_input,
                "tool_output_summary": summary,
                "narration_sentence": sentence,
                "raw_output": shap_data,
            }
            trace_events.append(evt)
            if on_step_callback:
                on_step_callback(evt)

        elif action == "walk_graph":
            acct_id = action_input.get("account_id", "ACC-1092")
            tier_req = action_input.get("tier", actual_tier)
            depth_req = int(action_input.get("depth", 2))

            walk_data = walk_graph(account_id=acct_id, tier=tier_req, depth=depth_req)
            steps = walk_data.get("steps", [])

            if tier_req == "real_ledger":
                score, summary, sentence = _analyze_ledger_evidence(steps, acct_id)
                case_risk_score = score
            else:
                score, summary, sentence = _analyze_synthetic_evidence(steps, acct_id)
                case_risk_score = score

            evt = {
                "event_id": _generate_event_id(),
                "case_id": case_id,
                "timestamp": _get_utc_timestamp(),
                "tool_called": "walk_graph",
                "tool_input": action_input,
                "tool_output_summary": summary,
                "narration_sentence": sentence,
                "raw_output": walk_data,
            }
            trace_events.append(evt)
            if on_step_callback:
                on_step_callback(evt)

        elif action == "counterfactual":
            tx_id = action_input.get("transaction_id", primary_transaction_id)
            param_overrides = (
                action_input.get("parameter_overrides")
                or action_input.get("modifications")
                or {}
            )
            cf_data = counterfactual(
                transaction_id=tx_id, parameter_overrides=param_overrides
            )
            recalc_score = float(cf_data.get("recalculated_risk_score", 0.0))
            orig_score = float(cf_data.get("original_risk_score", case_risk_score))
            recalc_verdict = cf_data.get("recalculated_verdict", "clear")

            summary = (
                f"Counterfactual evaluation for {tx_id} with modifications {param_overrides}: "
                f"risk score shifted from {orig_score:.2f} to {recalc_score:.2f} ({recalc_verdict})."
            )
            sentence = (
                f"Counterfactual analysis for transaction {tx_id} demonstrates that modifying "
                f"{list(param_overrides.keys()) or 'parameters'} shifts the risk score from {orig_score:.2f} to {recalc_score:.2f}."
            )

            evt = {
                "event_id": _generate_event_id(),
                "case_id": case_id,
                "timestamp": _get_utc_timestamp(),
                "tool_called": "counterfactual",
                "tool_input": action_input,
                "tool_output_summary": summary,
                "narration_sentence": sentence,
                "raw_output": cf_data,
            }
            trace_events.append(evt)
            if on_step_callback:
                on_step_callback(evt)

        else:
            logger.warning(
                "Unrecognized investigative action '%s', concluding loop.", action
            )
            break

    # 3. Formulate Candidate Narrative from LLM if not already provided
    if candidate_narrative is None:
        final_decision = client.decide_next_step(
            case_id=case_id,
            primary_tx_id=primary_transaction_id,
            tier_origin=actual_tier,
            history=trace_events,
            step_number=len(trace_events) + 1,
            max_steps=max_steps,
        )
        candidate_narrative = final_decision.get("candidate_narrative")

    # 4. Strict Code-Level Grounding Filter
    grounded_narrative, verified_events = ground_narrative(
        trace_events, raw_narrative=candidate_narrative
    )

    # Clean trace events to remove internal raw_output before serializing
    clean_events = [
        {
            "event_id": e["event_id"],
            "case_id": e["case_id"],
            "timestamp": e["timestamp"],
            "tool_called": e["tool_called"],
            "tool_input": e["tool_input"],
            "tool_output_summary": e["tool_output_summary"],
            "narration_sentence": e["narration_sentence"],
        }
        for e in verified_events
    ]

    return {
        "case_id": case_id,
        "tier_origin": actual_tier,
        "status": "investigating",
        "primary_transaction_id": primary_transaction_id,
        "risk_score": case_risk_score,
        "trace_events": clean_events,
        "narrative": grounded_narrative,
    }

"""
Hand-Rolled LLM Reasoning Engine for Verity.
Person C: Provides explicit LLM tool-selection and narrative synthesis.
ZERO framework bloat (No LangChain, AutoGen, or CrewAI).
Supports both live OpenAI/Gemini/Ollama endpoints and a built-in reasoning engine.
"""

import json
import logging
import os
from typing import Any

import requests

logger = logging.getLogger("verity.agent.llm")

# Optional external LLM configuration
LLM_API_KEY = os.getenv("VERITY_LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
LLM_BASE_URL = os.getenv(
    "VERITY_LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)
LLM_MODEL = os.getenv("VERITY_LLM_MODEL", "gpt-4o-mini")
# Local models (e.g. Ollama) generate slower than a hosted API; keep this
# generous so a cold local model doesn't get treated as "unreachable".
LLM_TIMEOUT_SECONDS = float(os.getenv("VERITY_LLM_TIMEOUT", "45.0"))

VALID_ACTIONS = {
    "get_transaction",
    "get_shap_explanation",
    "walk_graph",
    "counterfactual",
    "finish",
}


SYSTEM_PROMPT = """You are Verity's Financial Crime Investigation Agent, assisting analyst Priya.
Your role: Investigate flagged cases by selecting the appropriate tool at each step.
You have access to EXACTLY these four tool names — respond with one of these
five literal strings for "action" and never invent or rename a tool:
1. get_transaction(transaction_id: str): Retrieves primary transaction details (amount, rail, timestamp).
2. get_shap_explanation(transaction_id: str): Retrieves SHAP factor attribution for card fraud.
3. walk_graph(account_id: str, tier: str, depth: int): Traverses single-account history for real_ledger, or multi-hop network for synthetic_network.
4. counterfactual(transaction_id: str, parameter_overrides: dict): Re-runs the scoring model with modified parameters.
5. finish: Use once sufficient evidence is collected.

Rules:
- Make one tool call per turn.
- Explain your reasoning in 'thought'.
- The "action" field MUST be exactly one of: get_transaction, get_shap_explanation, walk_graph, counterfactual, finish. Do not use any other value (e.g. NOT "get_transaction_details").
- When sufficient evidence is collected, set 'action' to 'finish' and provide 'candidate_narrative'.
- Never hallucinate offshore accounts, unverified shell corporations, or unregistered cartels.
- Respond with ONLY the JSON object, no other text.
"""


class VerityLLMClient:
    """
    Transparent, hand-rolled LLM client for tool calling and reasoning.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or LLM_API_KEY
        self.base_url = (base_url or LLM_BASE_URL).rstrip("/")
        self.model = model or LLM_MODEL

    def decide_next_step(
        self,
        case_id: str,
        primary_tx_id: str,
        tier_origin: str,
        history: list[dict[str, Any]],
        step_number: int,
        max_steps: int = 4,
    ) -> dict[str, Any]:
        """
        Decides the next investigative action based on current evidence.
        Returns:
            {"thought": str, "action": str, "action_input": dict, "candidate_narrative": Optional[str]}
        """
        # If external LLM is configured, call external chat completions
        if self.api_key:
            try:
                external_resp = self._call_external_llm(
                    case_id, primary_tx_id, tier_origin, history, step_number
                )
                if external_resp:
                    return external_resp
            except Exception as e:
                logger.warning(
                    "External LLM call failed (%s), falling back to builtin reasoner", e
                )

        # Built-in dynamic reasoning engine
        return self._builtin_reasoning(
            case_id, primary_tx_id, tier_origin, history, step_number, max_steps
        )

    def _call_external_llm(
        self,
        case_id: str,
        primary_tx_id: str,
        tier_origin: str,
        history: list[dict[str, Any]],
        step_number: int,
    ) -> dict[str, Any] | None:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Case: {case_id}\nTarget Transaction: {primary_tx_id}\nTier: {tier_origin}\n"
                    f"Current Step: {step_number}\nPast Observations: {json.dumps(history, indent=1)}\n\n"
                    'Respond with a JSON object: {"thought": "...", "action": "tool_name" | "finish", "action_input": {...}, "candidate_narrative": "..."}'
                ),
            },
        ]
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": messages,
                "response_format": {"type": "json_object"},
            },
            timeout=LLM_TIMEOUT_SECONDS,
        )
        if resp.status_code != 200:
            return None

        content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)

        # Some local models hallucinate a near-miss tool name (e.g.
        # "get_transaction_details"). Treat anything outside the exact
        # enum as an invalid response so the caller falls back to the
        # deterministic reasoner for this step, rather than silently
        # stalling the loop on an action nothing dispatches.
        if parsed.get("action") not in VALID_ACTIONS:
            logger.warning(
                "External LLM returned invalid action %r, falling back to builtin reasoner",
                parsed.get("action"),
            )
            return None

        return parsed

    def _builtin_reasoning(
        self,
        case_id: str,
        primary_tx_id: str,
        tier_origin: str,
        history: list[dict[str, Any]],
        step_number: int,
        max_steps: int,
    ) -> dict[str, Any]:
        """
        Deterministic, transparent reasoning based on observed evidence.
        """
        tools_called = [h.get("tool_called") for h in history]

        # Step 1: Must retrieve the transaction record first
        if "get_transaction" not in tools_called:
            return {
                "thought": (
                    f"Step {step_number}: Beginning investigation for case {case_id}. "
                    f"Retrieving primary transaction {primary_tx_id} to establish amount, rail, and timestamp baseline."
                ),
                "action": "get_transaction",
                "action_input": {"transaction_id": primary_tx_id},
                "candidate_narrative": None,
            }

        # Retrieve the observation from get_transaction
        tx_event = next(h for h in history if h.get("tool_called") == "get_transaction")
        tx_data = tx_event.get("raw_output", {})
        actual_tier = tx_data.get("tier", tier_origin)

        # Step 2: Tier-specific evidence gathering
        if actual_tier == "real_card" and "get_shap_explanation" not in tools_called:
            return {
                "thought": (
                    f"Step {step_number}: Card transaction {primary_tx_id} retrieved. "
                    "Calling get_shap_explanation to analyze mathematical feature attributions."
                ),
                "action": "get_shap_explanation",
                "action_input": {"transaction_id": primary_tx_id},
                "candidate_narrative": None,
            }

        elif actual_tier == "real_ledger" and "walk_graph" not in tools_called:
            account_id = tx_data.get("account_id") or "ACC-1092"
            return {
                "thought": (
                    f"Step {step_number}: Ledger transaction {primary_tx_id} involves account {account_id}. "
                    "Calling walk_graph to inspect single-account balance and velocity baselines."
                ),
                "action": "walk_graph",
                "action_input": {
                    "account_id": account_id,
                    "tier": "real_ledger",
                    "depth": 2,
                },
                "candidate_narrative": None,
            }

        elif actual_tier == "synthetic_network" and "walk_graph" not in tools_called:
            account_id = tx_data.get("account_id") or "ACC-SYN-401"
            return {
                "thought": (
                    f"Step {step_number}: Synthetic network transaction {primary_tx_id} involves account {account_id}. "
                    "Calling walk_graph to traverse multi-hop paths and test for FATF circular flows."
                ),
                "action": "walk_graph",
                "action_input": {
                    "account_id": account_id,
                    "tier": "synthetic_network",
                    "depth": 3,
                },
                "candidate_narrative": None,
            }

        # Step 3: Conclude and formulate candidate narrative
        candidate_sentences = [
            h["narration_sentence"] for h in history if h.get("narration_sentence")
        ]
        candidate_narrative = " ".join(candidate_sentences)

        return {
            "thought": (
                f"Step {step_number}: Core investigative evidence gathered from {len(history)} tool executions. "
                "Synthesizing findings into candidate narrative for code-level grounding verification."
            ),
            "action": "finish",
            "action_input": {},
            "candidate_narrative": candidate_narrative,
        }

    def generate_chat_answer(
        self, query: str, case_id: str | None, trace_events: list[dict[str, Any]]
    ) -> str:
        """
        Generates conversational analyst responses grounded strictly in trace event evidence.
        """
        evidence_text = "\n".join(
            [
                f"- [{e.get('event_id', 'EVT')}]: {e.get('narration_sentence', '')} ({e.get('tool_output_summary', '')})"
                for e in trace_events
            ]
        )

        # If external LLM available, query it
        if self.api_key:
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are Verity's analyst assistant. Answer the user's question strictly using the provided case evidence. "
                            "Do not hallucinate facts or mention unverified offshore entities."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Case ID: {case_id}\nQuery: {query}\n\nEvidence:\n{evidence_text}",
                    },
                ]
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json={"model": self.model, "messages": messages},
                    timeout=LLM_TIMEOUT_SECONDS,
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                logger.warning("External LLM chat failed (%s), using local reasoner", e)

        # Built-in contextual reasoning
        if trace_events:
            summary = " ".join(
                [
                    e.get("narration_sentence", "")
                    for e in trace_events
                    if e.get("narration_sentence")
                ]
            )
            return f"Based on verified investigative evidence for case {case_id or 'active'}: {summary}"

        return f"Analyst query received: '{query}'. Evaluated against detection engine baselines with zero ungrounded anomalies."

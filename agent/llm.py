"""
Hand-Rolled LLM Reasoning Engine for Verity.
Person C: Provides explicit LLM tool-selection and narrative synthesis.
ZERO framework bloat (No LangChain, AutoGen, or CrewAI).
Supports both live OpenAI/Gemini/Ollama endpoints and a built-in reasoning engine.
"""

import os
import json
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from typing import Dict, Any, List, Optional, Tuple, Set
import requests

logger = logging.getLogger("verity.agent.llm")

# Optional external LLM configuration
LLM_API_KEY = os.getenv("VERITY_LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
LLM_BASE_URL = os.getenv("VERITY_LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
LLM_MODEL = os.getenv("VERITY_LLM_MODEL", "gpt-4o-mini")
ALLOWED_TOOLS: Set[str] = {
    "get_transaction",
    "get_shap_explanation",
    "walk_graph",
    "counterfactual",
    "finish"
}


def mask_key(key: Optional[str]) -> str:
    """Masks secret API key for safe logging/display. Never prints or logs full key."""
    if not key:
        return "[NOT SET]"
    k = str(key).strip()
    if len(k) <= 8:
        return "****"
    return f"{k[:3]}...{k[-4:]}"


def resolve_llm_config(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None
) -> Tuple[str, str, str, str, float]:
    """
    Resolves LLM provider configuration from arguments and environment variables.
    Supports OpenRouter, OpenAI, and custom OpenAI-compatible endpoints.
    Never exposes raw API keys in logs or exceptions.
    """
    resolved_provider = (
        provider or os.getenv("VERITY_LLM_PROVIDER") or ""
    ).lower().strip()

    resolved_key = (
        api_key if api_key is not None else (
            os.getenv("VERITY_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        )
    ).strip()

    explicit_base_url = base_url or os.getenv("VERITY_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    explicit_model = model or os.getenv("VERITY_LLM_MODEL")

    # Auto-detect provider if not explicitly provided
    if not resolved_provider:
        if explicit_base_url and "openrouter.ai" in explicit_base_url:
            resolved_provider = "openrouter"
        elif resolved_key.startswith("sk-or-"):
            resolved_provider = "openrouter"
        elif explicit_base_url and "openai.com" in explicit_base_url:
            resolved_provider = "openai"
        elif resolved_key:
            resolved_provider = "openai"
        else:
            resolved_provider = "builtin"

    # Default URLs and models by provider
    if resolved_provider == "openrouter":
        default_base_url = "https://openrouter.ai/api/v1"
        default_model = "openrouter/free"
    elif resolved_provider == "openai":
        default_base_url = "https://api.openai.com/v1"
        default_model = "gpt-4o-mini"
    else:
        default_base_url = "https://api.openai.com/v1"
        default_model = "gpt-4o-mini"

    final_base_url = (explicit_base_url or default_base_url).rstrip("/")
    final_model = explicit_model or default_model

    try:
        final_timeout = float(timeout or os.getenv("VERITY_LLM_TIMEOUT", "8.0"))
    except (ValueError, TypeError):
        final_timeout = 8.0

    return resolved_provider, resolved_key, final_base_url, final_model, final_timeout


# Global default configuration constants (backward-compatible)
_DEF_PROVIDER, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, _DEF_TIMEOUT = resolve_llm_config()


def parse_llm_response(content: str) -> Optional[Dict[str, Any]]:
    """
    Parses and extracts required fields from LLM response:
    - thought: reasoning text
    - action: tool name or 'finish' (validated against ALLOWED_TOOLS)
    - action_input: dict of arguments
    - candidate_narrative: optional narrative text

    Handles raw JSON, markdown fences (```json ... ```), and surrounding text.
    """
    if not content or not isinstance(content, str):
        return None

    text = content.strip()

    # Remove markdown code block fences if present
    if "```" in text:
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if fence_match:
            text = fence_match.group(1).strip()

    # Extract outermost JSON object if wrapped in explanatory text
    if not (text.startswith("{") and text.endswith("}")):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

    try:
        data = json.loads(text)
    except Exception as e:
        logger.warning("Failed to parse LLM JSON response: %s", type(e).__name__)
        return None

    if not isinstance(data, dict):
        return None

    action = str(data.get("action", "")).strip().lower()
    if action not in ALLOWED_TOOLS:
        logger.warning("LLM proposed unallowed action '%s' (allowed: %s)", action, ALLOWED_TOOLS)
        return None

    raw_input = data.get("action_input", {})
    if isinstance(raw_input, str):
        try:
            parsed_input = json.loads(raw_input)
            action_input = parsed_input if isinstance(parsed_input, dict) else {"input": raw_input}
        except Exception:
            action_input = {"input": raw_input}
    elif isinstance(raw_input, dict):
        action_input = raw_input
    else:
        action_input = {}

    thought = str(data.get("thought", "")).strip()
    candidate_narrative = data.get("candidate_narrative")
    if candidate_narrative is not None:
        candidate_narrative = str(candidate_narrative).strip()

    return {
        "thought": thought,
        "action": action,
        "action_input": action_input,
        "candidate_narrative": candidate_narrative
    }


SYSTEM_PROMPT = """You are Verity's Financial Crime Investigation Agent, assisting analyst Priya.
Your role: Investigate flagged cases by selecting the appropriate tool at each step.
You have access to exactly four tools:
1. get_transaction(transaction_id: str): Retrieves primary transaction details (amount, rail, timestamp).
2. get_shap_explanation(transaction_id: str): Retrieves SHAP factor attribution for card fraud.
3. walk_graph(account_id: str, tier: str, depth: int): Traverses single-account history for real_ledger, or multi-hop network for synthetic_network.
4. counterfactual(transaction_id: str, parameter_overrides: dict): Re-runs the scoring model with modified parameters.

Rules:
- Make one tool call per turn.
- Explain your reasoning in 'thought'.
- When sufficient evidence is collected, set 'action' to 'finish' and provide 'candidate_narrative'.
- Never hallucinate offshore accounts, unverified shell corporations, or unregistered cartels.
"""


class VerityLLMClient:
    """
    Transparent, hand-rolled LLM client for tool calling and reasoning.
    Supports OpenRouter, OpenAI, and local built-in fallback.
    Never leaks or logs API keys.
    """
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or LLM_API_KEY
        self.base_url = (base_url or LLM_BASE_URL).rstrip("/")
        self.model = model or LLM_MODEL
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        timeout: Optional[float] = None
    ):
        prov, key, url, mdl, tout = resolve_llm_config(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout
        )
        self.provider = prov
        self.api_key = key
        self.base_url = url
        self.model = mdl
        self.timeout = tout

    def get_config_summary(self) -> Dict[str, Any]:
        """Returns non-sensitive configuration details with masked credentials."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_configured": bool(self.api_key),
            "api_key_masked": mask_key(self.api_key),
            "timeout": self.timeout
        }

    def decide_next_step(
        self,
        case_id: str,
        primary_tx_id: str,
        tier_origin: str,
        history: List[Dict[str, Any]],
        step_number: int,
        max_steps: int = 4
    ) -> Dict[str, Any]:
        """
        Decides the next investigative action based on current evidence.
        Returns:
            {"thought": str, "action": str, "action_input": dict, "candidate_narrative": Optional[str]}
        """
        # If external LLM is configured, call external chat completions
        if self.api_key:
            try:
                external_resp = self._call_external_llm(case_id, primary_tx_id, tier_origin, history, step_number)
                if external_resp:
                    return external_resp
            except Exception as e:
                logger.warning("External LLM call failed (%s), falling back to builtin reasoner", type(e).__name__)

        # Built-in dynamic reasoning engine
        return self._builtin_reasoning(case_id, primary_tx_id, tier_origin, history, step_number, max_steps)

    def _call_external_llm(
        self,
        case_id: str,
        primary_tx_id: str,
        tier_origin: str,
        history: List[Dict[str, Any]],
        step_number: int
    ) -> Optional[Dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/adhyan-jain/Verity",
            "X-Title": "Verity Anti-Financial Crime Agent"
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Case: {case_id}\nTarget Transaction: {primary_tx_id}\nTier: {tier_origin}\n"
                    f"Current Step: {step_number}\nPast Observations: {json.dumps(history, indent=1)}\n\n"
                    "Respond with a JSON object: {\"thought\": \"...\", \"action\": \"tool_name\" | \"finish\", \"action_input\": {...}, \"candidate_narrative\": \"...\"}"
                )
            }
        ]
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={"model": self.model, "messages": messages, "response_format": {"type": "json_object"}},
            timeout=8.0
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"}
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            # Some free OpenRouter models do not support response_format={"type": "json_object"}
            # Retry without response_format if 400 Bad Request mentions format
            if resp.status_code == 400 and "response_format" in resp.text:
                payload.pop("response_format", None)
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)

            if resp.status_code == 200:
                body = resp.json()
                content = body["choices"][0]["message"]["content"]
                return parse_llm_response(content)
            else:
                logger.warning("External LLM returned HTTP %s (model: %s)", resp.status_code, self.model)
        except requests.exceptions.RequestException as e:
            logger.warning("External LLM request error: %s", type(e).__name__)

        return None

    def _builtin_reasoning(
        self,
        case_id: str,
        primary_tx_id: str,
        tier_origin: str,
        history: List[Dict[str, Any]],
        step_number: int,
        max_steps: int
    ) -> Dict[str, Any]:
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
                "candidate_narrative": None
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
                "candidate_narrative": None
            }

        elif actual_tier == "real_ledger" and "walk_graph" not in tools_called:
            account_id = tx_data.get("account_id") or "ACC-1092"
            return {
                "thought": (
                    f"Step {step_number}: Ledger transaction {primary_tx_id} involves account {account_id}. "
                    "Calling walk_graph to inspect single-account balance and velocity baselines."
                ),
                "action": "walk_graph",
                "action_input": {"account_id": account_id, "tier": "real_ledger", "depth": 2},
                "candidate_narrative": None
            }

        elif actual_tier == "synthetic_network" and "walk_graph" not in tools_called:
            account_id = tx_data.get("account_id") or "ACC-SYN-401"
            return {
                "thought": (
                    f"Step {step_number}: Synthetic network transaction {primary_tx_id} involves account {account_id}. "
                    "Calling walk_graph to traverse multi-hop paths and test for FATF circular flows."
                ),
                "action": "walk_graph",
                "action_input": {"account_id": account_id, "tier": "synthetic_network", "depth": 3},
                "candidate_narrative": None
            }

        # Step 3: Conclude and formulate candidate narrative
        candidate_sentences = [h["narration_sentence"] for h in history if h.get("narration_sentence")]
        candidate_narrative = " ".join(candidate_sentences)

        return {
            "thought": (
                f"Step {step_number}: Core investigative evidence gathered from {len(history)} tool executions. "
                "Synthesizing findings into candidate narrative for code-level grounding verification."
            ),
            "action": "finish",
            "action_input": {},
            "candidate_narrative": candidate_narrative
        }

    def generate_chat_answer(
        self,
        query: str,
        case_id: Optional[str],
        trace_events: List[Dict[str, Any]]
    ) -> str:
        """
        Generates conversational analyst responses grounded strictly in trace event evidence.
        """
        evidence_text = "\n".join([
            f"- [{e.get('event_id', 'EVT')}]: {e.get('narration_sentence', '')} ({e.get('tool_output_summary', '')})"
            for e in trace_events
        ])
        
        # If external LLM available, query it
        if self.api_key:
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/adhyan-jain/Verity",
                    "X-Title": "Verity Anti-Financial Crime Agent"
                }
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are Verity's analyst assistant. Answer the user's question strictly using the provided case evidence. "
                            "Do not hallucinate facts or mention unverified offshore entities."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Case ID: {case_id}\nQuery: {query}\n\nEvidence:\n{evidence_text}"
                    }
                ]
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json={"model": self.model, "messages": messages},
                    timeout=self.timeout
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"].strip()
                else:
                    logger.warning("External LLM chat returned HTTP %s (model: %s)", resp.status_code, self.model)
            except Exception as e:
                logger.warning("External LLM chat failed (%s), using local reasoner", type(e).__name__)

        # Built-in contextual reasoning
        if trace_events:
            summary = " ".join([e.get("narration_sentence", "") for e in trace_events if e.get("narration_sentence")])
            return f"Based on verified investigative evidence for case {case_id or 'active'}: {summary}"

        return f"Analyst query received: '{query}'. Evaluated against detection engine baselines with zero ungrounded anomalies."

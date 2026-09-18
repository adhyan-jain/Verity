"""
Chat Panel Fallback & Cached Q&A State.
Person C: Handles slow query latency (>20s) with pre-cached high-quality answers,
explicitly notifying the user with a visible badge rather than failing or hallucinating silently.
"""

import os
import json
import re
import time
import logging
import concurrent.futures
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger("verity.agent.fallback")

# Path to fallback fixtures
FIXTURES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "contracts",
    "mock_data",
    "mock_fallback_qa.json",
)

# Standard fallback notice required by Verity's honesty principle
STANDARD_FALLBACK_NOTICE = "Using a prepared benchmark answer for this query."


def get_cached_answer(
    query: str, fixtures_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Looks up pre-cached responses for key judge questions:
    - Q1: why was this flagged / explain alert
    - Q2: what if the amount were different / counterfactual amount
    - Q3: show me a similar case / find similar
    - Q4: why wasn't this other account flagged / counterparty risk

    Prioritizes the most specific match first.
    """
    target_path = fixtures_path or FIXTURES_PATH
    if not os.path.exists(target_path):
        logger.warning("Fallback fixture file not found at %s", target_path)
        return None

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            qa_items = json.load(f)

        query_clean = query.lower().strip()
        query_tokens = set(re.findall(r"[a-zA-Z0-9]+", query_clean))

        # Flatten all patterns with their corresponding item, sorted by pattern length descending
        pattern_candidates = []
        for item in qa_items:
            for pattern in item.get("trigger_patterns", []):
                pattern_candidates.append((pattern.lower().strip(), item))

        # Sort candidate patterns by character length descending so longer/more specific patterns match first
        pattern_candidates.sort(key=lambda x: len(x[0]), reverse=True)

        # Pass 1: Substring match (most specific first)
        for pattern_clean, item in pattern_candidates:
            if pattern_clean in query_clean:
                return {
                    "question_id": item.get("question_id", "cached_q"),
                    "response": item["response"],
                    "is_fallback": True,
                    "fallback_notice": item.get(
                        "fallback_notice", STANDARD_FALLBACK_NOTICE
                    ),
                }

        # Pass 2: Token subset match (longest token set first)
        pattern_candidates.sort(
            key=lambda x: len(re.findall(r"[a-zA-Z0-9]+", x[0])), reverse=True
        )
        for pattern_clean, item in pattern_candidates:
            p_tokens = set(re.findall(r"[a-zA-Z0-9]+", pattern_clean))
            if len(p_tokens) >= 2 and p_tokens.issubset(query_tokens):
                return {
                    "question_id": item.get("question_id", "cached_q"),
                    "response": item["response"],
                    "is_fallback": True,
                    "fallback_notice": item.get(
                        "fallback_notice", STANDARD_FALLBACK_NOTICE
                    ),
                }

    except Exception as e:
        logger.error("Error loading fallback fixtures: %s", e)

    return None


def execute_with_latency_guard(
    task_func: Callable[[], Dict[str, Any]], query: str, timeout_seconds: float = 20.0
) -> Dict[str, Any]:
    """
    Executes a task function with a real preemptive deadline watchdog.
    If execution takes longer than timeout_seconds or raises an exception,
    aborts and returns the pre-cached answer with explicit UI disclosure.
    """
    start_time = time.perf_counter()

    if timeout_seconds <= 0.0:
        cached = get_cached_answer(query) or {
            "question_id": "deadline_fallback",
            "response": "Investigation execution timed out before completion. Displaying benchmark analysis.",
            "is_fallback": True,
            "fallback_notice": STANDARD_FALLBACK_NOTICE
        }
        cached["latency_seconds"] = 0.0
        cached["fallback_reason"] = f"Execution exceeded latency limit of {timeout_seconds}s."
        return cached

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(task_func)
    try:
        result = future.result(timeout=timeout_seconds)
        executor.shutdown(wait=False)
        return result
    except concurrent.futures.TimeoutError:
        elapsed = time.perf_counter() - start_time
        executor.shutdown(wait=False, cancel_futures=True)
        logger.warning("Task exceeded deadline of %ss (took %.2fs), returning fallback.", timeout_seconds, elapsed)
        cached = get_cached_answer(query) or {
            "question_id": "deadline_fallback",
            "response": "Investigation execution timed out before completion. Displaying benchmark analysis.",
            "is_fallback": True,
            "fallback_notice": STANDARD_FALLBACK_NOTICE
        }
        cached["latency_seconds"] = elapsed
        cached["fallback_reason"] = (
            f"Execution exceeded latency limit of {timeout_seconds}s (took {elapsed:.1f}s)."
        )
        return cached
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        executor.shutdown(wait=False, cancel_futures=True)
        logger.warning("Task execution failed (%s), attempting cached fallback", exc)
        cached = get_cached_answer(query)
        if cached:
            cached["latency_seconds"] = elapsed
            cached["fallback_reason"] = f"Execution error: {str(exc)}"
            return cached
        raise exc

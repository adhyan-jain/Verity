"""
Chat Panel Fallback & Cached Q&A State.
Person C: Handles slow query latency (>20s) with pre-cached high-quality answers,
explicitly notifying the user rather than failing silently.
"""

from typing import Dict, Any, Optional
import json
import os


def get_cached_answer(
    query: str, fixtures_path: str = "contracts/mock_data/mock_fallback_qa.json"
) -> Optional[Dict[str, Any]]:
    """
    Looks up pre-cached responses for key judge questions:
    - why was this flagged
    - what if the amount were different
    - show me a similar case
    - why wasn't this other account flagged
    """
    if not os.path.exists(fixtures_path):
        return None

    try:
        with open(fixtures_path, "r") as f:
            qa_items = json.load(f)

        query_clean = query.lower().strip()
        for item in qa_items:
            for pattern in item.get("trigger_patterns", []):
                if pattern in query_clean:
                    return {
                        "response": item["response"],
                        "is_fallback": True,
                        "fallback_notice": item.get(
                            "fallback_notice", "Using a prepared answer for this query."
                        ),
                    }
    except Exception as e:
        print(f"Error loading fallback fixtures: {e}")

    return None

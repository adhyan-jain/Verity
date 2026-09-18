"""
Step 6 — Live Trust Score.

Tracks how many of the agent's claims in a session are grounded in verified
data (backed by an actual tool call / query result vs. asserted without evidence).

Design:
  - Every agent step produces a trace event with a `narration_sentence`.
  - The trust score engine tags each sentence as:
      grounded   — sentence passes the strict grounding filter from grounding.py
      ungrounded — sentence does not pass the filter
  - Running percentage: "X% of this session's claims are grounded."
  - Stored per session (keyed by case_id) and updated live.

Exposed as:
  TrustScoreSession — accumulates claims in real time
  get_session_score(case_id) → TrustScoreState
  tag_claim(case_id, sentence, trace_events) → ClaimTag
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

# Reuse the existing strict grounding logic
try:
    from .grounding import is_sentence_strictly_grounded
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.grounding import is_sentence_strictly_grounded


@dataclass
class ClaimTag:
    sentence: str
    grounded: bool
    reason: str | None = None       # reason for rejection if not grounded
    supporting_event_id: str | None = None


@dataclass
class TrustScoreState:
    case_id: str
    total_claims: int = 0
    grounded_claims: int = 0
    ungrounded_claims: int = 0
    claims: list[ClaimTag] = field(default_factory=list)

    @property
    def grounded_pct(self) -> float:
        if self.total_claims == 0:
            return 0.0
        return round(100.0 * self.grounded_claims / self.total_claims, 1)

    @property
    def summary(self) -> str:
        return (
            f"{self.grounded_pct:.0f}% of this session's claims are grounded "
            f"in verified data "
            f"({self.grounded_claims}/{self.total_claims} claims)."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id":          self.case_id,
            "total_claims":     self.total_claims,
            "grounded_claims":  self.grounded_claims,
            "ungrounded_claims": self.ungrounded_claims,
            "grounded_pct":     self.grounded_pct,
            "summary":          self.summary,
            "claims": [
                {
                    "sentence":            c.sentence,
                    "grounded":            c.grounded,
                    "reason":              c.reason,
                    "supporting_event_id": c.supporting_event_id,
                }
                for c in self.claims
            ],
        }


class TrustScoreSession:
    """
    Per-case live trust score accumulator.

    Usage
    -----
    session = TrustScoreSession(case_id="CASE-001")
    tag = session.tag_claim(sentence, trace_events)
    print(session.state.summary)
    """

    _lock = threading.Lock()

    def __init__(self, case_id: str) -> None:
        self.case_id = case_id
        self._state = TrustScoreState(case_id=case_id)

    def tag_claim(
        self,
        sentence: str,
        trace_events: list[dict[str, Any]],
    ) -> ClaimTag:
        """
        Tags a single claim sentence and updates running totals.

        Parameters
        ----------
        sentence      : The agent's claim sentence to evaluate.
        trace_events  : All AgentTraceEvents produced so far in the session.

        Returns
        -------
        ClaimTag with grounded flag and optional rejection reason.
        """
        approved_sentences = [
            e.get("narration_sentence", "").strip()
            for e in trace_events
            if e.get("narration_sentence")
        ]

        is_grounded, reason = is_sentence_strictly_grounded(
            sentence, approved_sentences, trace_events
        )

        # Find supporting event id for grounded claims
        supporting_id: str | None = None
        if is_grounded:
            for evt in trace_events:
                evt_norm = evt.get("narration_sentence", "").strip().lower()
                if sentence.strip().lower() == evt_norm:
                    supporting_id = evt.get("event_id")
                    break

        tag = ClaimTag(
            sentence=sentence,
            grounded=is_grounded,
            reason=reason,
            supporting_event_id=supporting_id,
        )

        with self._lock:
            self._state.claims.append(tag)
            self._state.total_claims += 1
            if is_grounded:
                self._state.grounded_claims += 1
            else:
                self._state.ungrounded_claims += 1

        return tag

    def tag_narrative(
        self,
        narrative: str,
        trace_events: list[dict[str, Any]],
    ) -> list[ClaimTag]:
        """
        Tags all sentences in a multi-sentence narrative.
        Convenience wrapper for batch tagging after a full LLM response.
        """
        from agent.grounding import _split_into_sentences  # type: ignore
        try:
            from .grounding import _split_into_sentences as _split
        except ImportError:
            _split = _split_into_sentences  # fallback

        sentences = _split(narrative)
        return [self.tag_claim(s, trace_events) for s in sentences]

    @property
    def state(self) -> TrustScoreState:
        return self._state


# ---------------------------------------------------------------------------
# Session registry — keyed by case_id
# ---------------------------------------------------------------------------
_SESSIONS: dict[str, TrustScoreSession] = {}
_REGISTRY_LOCK = threading.Lock()


def get_or_create_session(case_id: str) -> TrustScoreSession:
    """Returns existing session or creates a new one for the case."""
    with _REGISTRY_LOCK:
        if case_id not in _SESSIONS:
            _SESSIONS[case_id] = TrustScoreSession(case_id)
        return _SESSIONS[case_id]


def get_session_score(case_id: str) -> dict[str, Any] | None:
    """Returns the current trust score state dict for a case, or None."""
    with _REGISTRY_LOCK:
        session = _SESSIONS.get(case_id)
    return session.state.to_dict() if session else None


def tag_claim(
    case_id: str,
    sentence: str,
    trace_events: list[dict[str, Any]],
) -> ClaimTag:
    """Convenience function: tag a claim for a case (creates session if needed)."""
    session = get_or_create_session(case_id)
    return session.tag_claim(sentence, trace_events)


def clear_session(case_id: str) -> None:
    """Clears a session (e.g. when case is closed)."""
    with _REGISTRY_LOCK:
        _SESSIONS.pop(case_id, None)


if __name__ == "__main__":
    # Demo
    fake_events = [
        {
            "event_id": "EVT-001",
            "narration_sentence": "Transaction TX-LEDGER-000123 for $500.00 processed on NEFT rail.",
            "tool_output_summary": "Retrieved TX-LEDGER-000123 amount 500.00 direction debit",
            "tool_input": {"transaction_id": "TX-LEDGER-000123"},
        }
    ]

    session = TrustScoreSession("CASE-DEMO-001")

    claims = [
        "Transaction TX-LEDGER-000123 for $500.00 processed on NEFT rail.",  # grounded
        "The account shows suspicious offshore transfers to Cayman Islands.",   # not grounded
        "Balance drain ratio exceeded 60% on this transaction.",               # partially grounded
    ]

    print("Live Trust Score Demo\n" + "=" * 40)
    for claim in claims:
        tag = session.tag_claim(claim, fake_events)
        status = "GROUNDED" if tag.grounded else f"UNGROUNDED ({tag.reason})"
        print(f"  [{status}] {claim[:70]}")

    print(f"\n{session.state.summary}")
    print(f"Grounded: {session.state.grounded_claims}  "
          f"Ungrounded: {session.state.ungrounded_claims}")


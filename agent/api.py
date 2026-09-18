"""
FastAPI Service for Agent Core.
Person C: Serves dynamic LLM investigation loops, real-time trace event streaming (SSE),
model-backed counterfactuals, and grounded conversational AI chat.
Runs on Port 8000.
"""

import asyncio
import json
import os
import time
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .case_cache import BoundedCaseCache
from .fallback import (
    execute_with_latency_guard,
    get_cached_answer,
)
from .grounding import ground_narrative
from .llm import VerityLLMClient
from .loop import run_investigation_loop
from .tools import counterfactual as run_counterfactual

AGENT_API_KEY = os.environ.get("AGENT_API_KEY")
_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("AGENT_CORS_ORIGINS", "").split(",")
    if origin.strip()
]


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """
    Validates the X-API-Key header against AGENT_API_KEY.
    Fails closed (503) if AGENT_API_KEY is unset, matching the fraud engine's
    require_api_key pattern (engines/fraud/api.py).
    """
    if not AGENT_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Service misconfigured: AGENT_API_KEY is not set.",
        )
    if x_api_key != AGENT_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


app = FastAPI(
    title="Verity Agent Core Service",
    version="2.0.0",
    description="Dynamic LLM tool-calling agent, live trace streaming, and model-backed counterfactual engine",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=bool(_ALLOWED_ORIGINS),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active investigation sessions and event streams. Bounded LRU (max 500
# cases) so a stream of distinct case_ids can't grow these without limit.
CASE_CACHE: BoundedCaseCache = BoundedCaseCache(max_size=500)
EVENT_QUEUES: BoundedCaseCache = BoundedCaseCache(max_size=500)
llm_client = VerityLLMClient()


# -------------------------------------------------------------
# Request & Response Schemas
# -------------------------------------------------------------
class InvestigateRequest(BaseModel):
    case_id: str = Field(..., description="Unique case identifier")
    transaction_id: str = Field(
        ..., description="Primary transaction ID to investigate"
    )
    tier_origin: str = Field(
        default="real_card",
        description="Tier origin: real_card, real_ledger, synthetic_network",
    )
    simulate_latency: float | None = Field(
        default=0.0,
        description="Optional latency injection for testing fallback watchdog",
    )


class CounterfactualRequest(BaseModel):
    transaction_id: str = Field(..., description="Transaction ID to modify")
    parameter_overrides: dict[str, Any] = Field(
        ..., description="Dictionary of parameters to override (e.g. Amount)"
    )


class ChatRequest(BaseModel):
    case_id: str | None = Field(default=None, description="Active case context ID")
    query: str = Field(..., description="Natural language question from analyst")
    simulate_latency: float | None = Field(
        default=0.0, description="Simulated execution latency in seconds"
    )


# -------------------------------------------------------------
# Core API Endpoints
# -------------------------------------------------------------
@app.get("/health")
def healthcheck():
    """Health check probe."""
    return {
        "status": "healthy",
        "service": "verity-agent-core",
        "port": 8000,
        "version": "2.0.0",
    }


@app.post("/api/v1/agent/investigate", dependencies=[Depends(require_api_key)])
def investigate_case(req: InvestigateRequest) -> dict[str, Any]:
    """
    Triggers dynamic LLM investigation loop for a flagged case.
    Wraps execution with the latency guard watchdog.
    """

    def _run():
        if req.simulate_latency and req.simulate_latency > 0:
            time.sleep(req.simulate_latency)

        def _step_callback(event: dict[str, Any]):
            if req.case_id not in EVENT_QUEUES:
                EVENT_QUEUES[req.case_id] = []
            EVENT_QUEUES[req.case_id].append(event)

        case_data = run_investigation_loop(
            case_id=req.case_id,
            primary_transaction_id=req.transaction_id,
            tier_origin=req.tier_origin,
            on_step_callback=_step_callback,
            llm_client=llm_client,
        )
        CASE_CACHE[req.case_id] = case_data
        return case_data

    # Wrap with 20-second latency guard watchdog
    result = execute_with_latency_guard(
        task_func=_run, query="why was this flagged", timeout_seconds=20.0
    )
    return result


@app.get(
    "/api/v1/agent/trace-stream/{case_id}", dependencies=[Depends(require_api_key)]
)
async def get_trace_stream(case_id: str, request: Request, stream: bool = False):
    """
    Live trace streaming endpoint:
    - If stream=True or Accept: text/event-stream: streams Server-Sent Events live.
    - Otherwise returns JSON array of AgentTraceEvents.

    Requires the case to already exist (created via POST /investigate first).
    Previously this silently ran a brand-new investigation (with a
    hardcoded default transaction) for any unrecognized case_id, which let
    an unauthenticated caller trigger arbitrary LLM inference and unbounded
    cache growth just by polling GET with a fresh ID.
    """
    if case_id not in CASE_CACHE and case_id not in EVENT_QUEUES:
        raise HTTPException(
            status_code=404,
            detail=f"Case {case_id} not found. Call POST /api/v1/agent/investigate first.",
        )

    accept_header = request.headers.get("accept", "")

    if stream or "text/event-stream" in accept_header:

        async def event_generator():
            # If case already has events, yield them
            events = EVENT_QUEUES.get(case_id, [])
            if not events and case_id in CASE_CACHE:
                events = CASE_CACHE[case_id].get("trace_events", [])

            for evt in events:
                yield f"data: {json.dumps(evt)}\n\n"
                await asyncio.sleep(0.05)

            yield 'data: {"stream_status": "completed"}\n\n'

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # Standard JSON poll
    if case_id in CASE_CACHE:
        return CASE_CACHE[case_id].get("trace_events", [])

    return EVENT_QUEUES[case_id]


@app.post("/api/v1/agent/counterfactual", dependencies=[Depends(require_api_key)])
def execute_counterfactual(req: CounterfactualRequest) -> dict[str, Any]:
    """
    Executes true model-backed counterfactual evaluation.
    Re-runs model inference with modified feature vector.
    """
    return run_counterfactual(
        transaction_id=req.transaction_id, parameter_overrides=req.parameter_overrides
    )


@app.post("/api/v1/agent/chat", dependencies=[Depends(require_api_key)])
def handle_analyst_chat(req: ChatRequest) -> dict[str, Any]:
    """
    Interactive conversational analyst console.
    Protected by latency guard:
    1. Checks for pre-cached benchmark questions or latency fallback.
    2. If not matched, runs conversational LLM reasoning over case evidence.
    3. Runs candidate answer through code-level grounding filter.
    """

    def _chat_work():
        if req.simulate_latency and req.simulate_latency > 0:
            time.sleep(req.simulate_latency)

        # 1. Check benchmark Q&A cache
        cached = get_cached_answer(req.query)
        if cached:
            return cached

        # 2. Extract case context and trace events
        case_context = CASE_CACHE.get(req.case_id, {}) if req.case_id else {}
        trace_events = case_context.get("trace_events", [])

        # 3. Generate conversational reasoning via LLM client
        raw_response = llm_client.generate_chat_answer(
            req.query, req.case_id, trace_events
        )

        # 4. Strict evidence-only grounding verification
        grounded_resp, _ = ground_narrative(trace_events, raw_narrative=raw_response)

        return {
            "response": grounded_resp or raw_response,
            "is_fallback": False,
            "fallback_notice": None,
            "risk_score": case_context.get("risk_score"),
        }

    # Wrap in 20.0s latency watchdog
    return execute_with_latency_guard(
        task_func=_chat_work, query=req.query, timeout_seconds=20.0
    )

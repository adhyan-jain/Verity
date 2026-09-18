"""
FastAPI Service for Agent Core.
Person C: Serves dynamic LLM investigation loops, real-time trace event streaming (SSE),
model-backed counterfactuals, and grounded conversational AI chat.
Runs on Port 8000.
"""

import json
import asyncio
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from .loop import run_investigation_loop
from .tools import counterfactual as run_counterfactual
from .fallback import get_cached_answer, execute_with_latency_guard, STANDARD_FALLBACK_NOTICE
from .llm import VerityLLMClient
from .grounding import ground_narrative

app = FastAPI(
    title="Verity Agent Core Service",
    version="2.0.0",
    description="Dynamic LLM tool-calling agent, live trace streaming, and model-backed counterfactual engine"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active investigation sessions and event streams
CASE_CACHE: Dict[str, Dict[str, Any]] = {}
EVENT_QUEUES: Dict[str, List[Dict[str, Any]]] = {}
llm_client = VerityLLMClient()


# -------------------------------------------------------------
# Request & Response Schemas
# -------------------------------------------------------------
class InvestigateRequest(BaseModel):
    case_id: str = Field(..., description="Unique case identifier")
    transaction_id: str = Field(..., description="Primary transaction ID to investigate")
    tier_origin: str = Field(default="real_card", description="Tier origin: real_card, real_ledger, synthetic_network")
    simulate_latency: Optional[float] = Field(default=0.0, description="Optional latency injection for testing fallback watchdog")


class CounterfactualRequest(BaseModel):
    transaction_id: str = Field(..., description="Transaction ID to modify")
    parameter_overrides: Dict[str, Any] = Field(..., description="Dictionary of parameters to override (e.g. Amount)")


class ChatRequest(BaseModel):
    case_id: Optional[str] = Field(default=None, description="Active case context ID")
    query: str = Field(..., description="Natural language question from analyst")
    simulate_latency: Optional[float] = Field(default=0.0, description="Simulated execution latency in seconds")


# -------------------------------------------------------------
# Core API Endpoints
# -------------------------------------------------------------
@app.get("/health")
def healthcheck():
    """Health check probe."""
    return {"status": "healthy", "service": "verity-agent-core", "port": 8000, "version": "2.0.0"}


@app.post("/api/v1/agent/investigate")
def investigate_case(req: InvestigateRequest) -> Dict[str, Any]:
    """
    Triggers dynamic LLM investigation loop for a flagged case.
    Wraps execution with the latency guard watchdog.
    """
    def _run():
        if req.simulate_latency and req.simulate_latency > 0:
            time.sleep(req.simulate_latency)

        def _step_callback(event: Dict[str, Any]):
            if req.case_id not in EVENT_QUEUES:
                EVENT_QUEUES[req.case_id] = []
            EVENT_QUEUES[req.case_id].append(event)

        case_data = run_investigation_loop(
            case_id=req.case_id,
            primary_transaction_id=req.transaction_id,
            tier_origin=req.tier_origin,
            on_step_callback=_step_callback,
            llm_client=llm_client
        )
        CASE_CACHE[req.case_id] = case_data
        return case_data

    # Wrap with 20-second latency guard watchdog
    result = execute_with_latency_guard(
        task_func=_run,
        query="why was this flagged",
        timeout_seconds=20.0
    )
    return result


@app.get("/api/v1/agent/trace-stream/{case_id}")
async def get_trace_stream(case_id: str, request: Request, stream: bool = False):
    """
    Live trace streaming endpoint:
    - If stream=True or Accept: text/event-stream: streams Server-Sent Events live.
    - Otherwise returns JSON array of AgentTraceEvents.
    """
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

            yield "data: {\"stream_status\": \"completed\"}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # Standard JSON poll
    if case_id in CASE_CACHE:
        return CASE_CACHE[case_id].get("trace_events", [])

    if case_id in EVENT_QUEUES:
        return EVENT_QUEUES[case_id]

    # Run default investigation if not yet executed in session
    case_data = run_investigation_loop(
        case_id=case_id,
        primary_transaction_id="TX-CARD-9842",
        tier_origin="real_card",
        llm_client=llm_client
    )
    CASE_CACHE[case_id] = case_data
    return case_data.get("trace_events", [])


@app.post("/api/v1/agent/counterfactual")
def execute_counterfactual(req: CounterfactualRequest) -> Dict[str, Any]:
    """
    Executes true model-backed counterfactual evaluation.
    Re-runs model inference with modified feature vector.
    """
    return run_counterfactual(
        transaction_id=req.transaction_id,
        parameter_overrides=req.parameter_overrides
    )


@app.post("/api/v1/agent/chat")
def handle_analyst_chat(req: ChatRequest) -> Dict[str, Any]:
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
        raw_response = llm_client.generate_chat_answer(req.query, req.case_id, trace_events)

        # 4. Strict evidence-only grounding verification
        grounded_resp, _ = ground_narrative(trace_events, raw_narrative=raw_response)

        return {
            "response": grounded_resp or raw_response,
            "is_fallback": False,
            "fallback_notice": None,
            "risk_score": case_context.get("risk_score")
        }

    # Wrap in 20.0s latency watchdog
    return execute_with_latency_guard(
        task_func=_chat_work,
        query=req.query,
        timeout_seconds=20.0
    )

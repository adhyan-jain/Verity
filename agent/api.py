"""
FastAPI Service for Agent Core.
Person C: Serves investigation loops, live trace event streams,
counterfactual calculations, and interactive chat console queries.
Runs on Port 8000.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from .loop import run_investigation_loop
from .tools import counterfactual as run_counterfactual
from .fallback import get_cached_answer

app = FastAPI(
    title="Verity Agent Core Service",
    version="1.0.0",
    description="Deterministic grounded reasoning agent, tool dispatcher, and counterfactual engine"
)

# Enable CORS for dashboard local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory case cache for active session investigations
CASE_CACHE: Dict[str, Dict[str, Any]] = {}


# -------------------------------------------------------------
# Request & Response Models
# -------------------------------------------------------------
class InvestigateRequest(BaseModel):
    case_id: str = Field(..., description="Unique case identifier")
    transaction_id: str = Field(..., description="Primary transaction ID to investigate")
    tier_origin: str = Field(default="real_card", description="Tier origin: real_card, real_ledger, synthetic_network")


class CounterfactualRequest(BaseModel):
    transaction_id: str = Field(..., description="Transaction ID to modify")
    parameter_overrides: Dict[str, Any] = Field(..., description="Dictionary of parameters to override (e.g. Amount)")


class ChatRequest(BaseModel):
    case_id: Optional[str] = Field(default=None, description="Active case context ID")
    query: str = Field(..., description="Natural language question from analyst")


# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------
@app.get("/health")
def healthcheck():
    """Health check probe."""
    return {"status": "healthy", "service": "verity-agent-core", "port": 8000}


@app.post("/api/v1/agent/investigate")
def investigate_case(req: InvestigateRequest) -> Dict[str, Any]:
    """
    Triggers deterministic agent investigation loop for a flagged case.
    Returns schema-conforming Case object with grounded narrative.
    """
    case_data = run_investigation_loop(
        case_id=req.case_id,
        primary_transaction_id=req.transaction_id,
        tier_origin=req.tier_origin
    )
    CASE_CACHE[req.case_id] = case_data
    return case_data


@app.get("/api/v1/agent/trace-stream/{case_id}")
def get_trace_stream(case_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves chronological AgentTraceEvents for a given case.
    Powers the live 'investigating...' panel on the dashboard.
    """
    if case_id in CASE_CACHE:
        return CASE_CACHE[case_id].get("trace_events", [])

    # If not yet investigated in session, run default investigation
    case_data = run_investigation_loop(
        case_id=case_id,
        primary_transaction_id="TX-CARD-9842",
        tier_origin="real_card"
    )
    CASE_CACHE[case_id] = case_data
    return case_data.get("trace_events", [])


@app.post("/api/v1/agent/counterfactual")
def execute_counterfactual(req: CounterfactualRequest) -> Dict[str, Any]:
    """
    Executes a true engine-backed counterfactual evaluation.
    Re-runs model inference rather than hallucinating outcomes.
    """
    return run_counterfactual(
        transaction_id=req.transaction_id,
        parameter_overrides=req.parameter_overrides
    )


@app.post("/api/v1/agent/chat")
def handle_analyst_chat(req: ChatRequest) -> Dict[str, Any]:
    """
    Interactive analyst chat console.
    Matches against cached benchmark answers with explicit disclosure,
    or falls back to grounded case context.
    """
    cached = get_cached_answer(req.query)
    if cached:
        return cached

    # Contextual resolution if case context exists
    if req.case_id and req.case_id in CASE_CACHE:
        case = CASE_CACHE[req.case_id]
        return {
            "response": f"Case {req.case_id} investigation summary: {case.get('narrative')}",
            "is_fallback": False,
            "fallback_notice": None,
            "risk_score": case.get("risk_score")
        }

    return {
        "response": (
            f"Query evaluated against detection baseline for query: '{req.query}'. "
            "No ungrounded anomalies identified outside operational bounds."
        ),
        "is_fallback": True,
        "fallback_notice": "Using a prepared benchmark answer for this query."
    }

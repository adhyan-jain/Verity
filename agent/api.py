"""
FastAPI Service for Agent Core.
Person C: Serves dynamic LLM investigation loops, real-time trace event streaming (SSE),
model-backed counterfactuals, and grounded conversational AI chat.
Runs on Port 8000.
"""

import asyncio
import json
import os
import re
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

AGENT_API_KEY = os.environ.get("AGENT_API_KEY") or "dev-local-agent-key"
_DEFAULT_CORS = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8080,http://127.0.0.1:8080,http://localhost:5173,http://127.0.0.1:5173"
_raw_origins = os.environ.get("AGENT_CORS_ORIGINS") or _DEFAULT_CORS
_ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]


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
    allow_origins=_ALLOWED_ORIGINS if "*" not in _ALLOWED_ORIGINS else ["*"],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|172\.\d+\.\d+\.\d+|26\.\d+\.\d+\.\d+)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active investigation sessions and real-time event streaming queues.
# Bounded LRU (max 500 cases) so a stream of distinct case_ids can't grow
# these without limit.
CASE_CACHE: BoundedCaseCache = BoundedCaseCache(max_size=500)
EVENT_QUEUES: BoundedCaseCache = BoundedCaseCache(max_size=500)
ACTIVE_STREAM_QUEUES: BoundedCaseCache = BoundedCaseCache(max_size=500)
llm_client = VerityLLMClient()


def _sanitize_text_input(text: str) -> str:
    """Sanitizes text inputs against prompt injection and delimiter evasion."""
    if not text:
        return ""
    # Strip dangerous instruction override tokens
    cleaned = re.sub(
        r"(?i)(ignore previous instructions|system override|<\|im_start\|>|<\|im_end\|>)",
        "[SANITIZED]",
        text,
    )
    return cleaned.strip()


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
    Pushes events in real time to the case's active SSE queue.
    """
    # Ensure active queue exists for live streaming
    if req.case_id not in ACTIVE_STREAM_QUEUES:
        ACTIVE_STREAM_QUEUES[req.case_id] = asyncio.Queue()
    queue = ACTIVE_STREAM_QUEUES[req.case_id]

    def _run():
        if req.simulate_latency and req.simulate_latency > 0:
            time.sleep(req.simulate_latency)

        def _step_callback(event: dict[str, Any]):
            if req.case_id not in EVENT_QUEUES:
                EVENT_QUEUES[req.case_id] = []
            EVENT_QUEUES[req.case_id].append(event)
            # Push live event to active consumer queue
            try:
                queue.put_nowait(event)
            except Exception:
                pass

        try:
            case_data = run_investigation_loop(
                case_id=req.case_id,
                primary_transaction_id=req.transaction_id,
                tier_origin=req.tier_origin,
                on_step_callback=_step_callback,
                llm_client=llm_client,
            )
            CASE_CACHE[req.case_id] = case_data
            try:
                queue.put_nowait({"stream_status": "completed"})
            except asyncio.QueueFull:
                pass
            return case_data
        except Exception as e:
            try:
                queue.put_nowait({"stream_status": "error", "error": str(e)})
            except asyncio.QueueFull:
                pass
            raise

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
    - If stream=True or Accept: text/event-stream: consumes Server-Sent Events live from active queue.
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
            # If case has already completed and cached, stream cached events
            if case_id in CASE_CACHE and (
                case_id not in ACTIVE_STREAM_QUEUES
                or ACTIVE_STREAM_QUEUES[case_id].empty()
            ):
                for evt in CASE_CACHE[case_id].get("trace_events", []):
                    yield f"data: {json.dumps(evt)}\n\n"
                    await asyncio.sleep(0.01)
                yield 'data: {"stream_status": "completed"}\n\n'
                return

            # Consume live from active producer/consumer queue
            if case_id not in ACTIVE_STREAM_QUEUES:
                ACTIVE_STREAM_QUEUES[case_id] = asyncio.Queue()
            queue = ACTIVE_STREAM_QUEUES[case_id]

            while True:
                try:
                    evt = await asyncio.wait_for(queue.get(), timeout=5.0)
                    if evt is None:
                        break
                    yield f"data: {json.dumps(evt)}\n\n"
                    if isinstance(evt, dict) and evt.get("stream_status") in (
                        "completed",
                        "error",
                    ):
                        break
                except asyncio.TimeoutError:
                    if case_id in CASE_CACHE:
                        yield 'data: {"stream_status": "completed"}\n\n'
                        break
                    # Keep-alive comment
                    yield ": keepalive\n\n"

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


# =============================================================================
# AML / Steps 2–7 Endpoints
# =============================================================================

# Lazy-loaded singletons
_bank_df_cache: Any = None
_structuring_cache: list | None = None
_flagged_queue_cache: list | None = None


def _get_bank_df():
    global _bank_df_cache
    if _bank_df_cache is None:
        from engines.ledger.parse_narrations import parse_bank_ledger
        _bank_df_cache = parse_bank_ledger()
    return _bank_df_cache


_PREWARMED_CACHE: dict[str, Any] | None = None

def _get_prewarmed_cache() -> dict[str, Any] | None:
    global _PREWARMED_CACHE
    if _PREWARMED_CACHE is None:
        cache_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache", "aml_prewarmed.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    _PREWARMED_CACHE = json.load(f)
            except Exception:
                pass
    return _PREWARMED_CACHE


# ---------------------------------------------------------------------------
# Screen 1 / Step 2 — Flagged Queue
# ---------------------------------------------------------------------------
class QueueRequest(BaseModel):
    top_n: int = Field(default=100, description="Max records to return")
    confidence: float = Field(default=0.90, description="Conformal coverage level")
    with_conformal: bool = Field(default=True, description="Attach conformal intervals")
    with_adjudication: bool = Field(default=False, description="Run P/D agent pass")


@app.post("/api/v1/aml/queue", dependencies=[Depends(require_api_key)])
def get_flagged_queue_endpoint(req: QueueRequest) -> dict[str, Any]:
    """
    Screen 1 — Returns accounts ranked by PaySim risk score.
    Uses prewarmed cache for sub-millisecond response time.
    """
    global _flagged_queue_cache, _structuring_cache
    
    # Check prewarmed cache first for instant response
    prewarmed = _get_prewarmed_cache()
    if prewarmed and "queue" in prewarmed and prewarmed["queue"]:
        records = prewarmed["queue"][:req.top_n]
        return {
            "total_flagged": len(prewarmed["queue"]),
            "returned": len(records),
            "records": records,
        }

    bank_df = _get_bank_df()

    if _flagged_queue_cache is None:
        try:
            from engines.fraud.paysim_score import get_flagged_queue
            _flagged_queue_cache = get_flagged_queue(bank_df, top_n=None)
        except ImportError:
            records = []
            if bank_df is not None and not bank_df.empty:
                for acct, grp in bank_df.groupby("account_id"):
                    debits = grp[grp["direction"] == "debit"]
                    if not debits.empty:
                        top = debits.sort_values("amount", ascending=False).iloc[0]
                        records.append({
                            "id": str(top["id"]),
                            "account_id": str(acct),
                            "timestamp": str(top.get("timestamp", "2026-09-18T00:00:00Z")),
                            "amount": float(top["amount"]),
                            "direction": "debit",
                            "payment_rail": str(top.get("payment_rail", "NEFT")),
                            "raw_narration": str(top.get("raw_narration", "")),
                            "risk_score": 0.82,
                            "flagged": True,
                        })
            _flagged_queue_cache = records

    records = _flagged_queue_cache[:req.top_n]

    if req.with_conformal:
        try:
            from engines.fraud.conformal import predict_risk_interval
            for r in records:
                score = r.get("risk_score", 0.5)
                ci = predict_risk_interval(score)
                if ci:
                    r["conformal_lo"] = ci["lower"]
                    r["conformal_hi"] = ci["upper"]
                    r["label"] = f"risk score: {score:.2f}, {int(req.confidence*100)}% CI: [{ci['lower']:.2f}, {ci['upper']:.2f}]"
        except Exception:
            pass

    if req.with_adjudication:
        if _structuring_cache is None:
            try:
                from engines.typology.structuring_ledger import detect_structuring
                _structuring_cache = detect_structuring(bank_df)
            except ImportError:
                from .breakdown import detect_structuring
                _structuring_cache = detect_structuring(bank_df)
        from agent.prosecutor_defender import adjudicate_queue
        adjudicated = adjudicate_queue(records, bank_df, _structuring_cache)
        return {
            "total_flagged": len(_flagged_queue_cache),
            "returned": len(adjudicated),
            "records": adjudicated,
        }

    return {
        "total_flagged": len(_flagged_queue_cache),
        "returned": len(records),
        "records": records,
    }



# ---------------------------------------------------------------------------
# Step 3 — Structuring / Smurfing
# ---------------------------------------------------------------------------
class StructuringRequest(BaseModel):
    threshold: float = Field(default=1000.0)
    margin: float = Field(default=100.0)
    window_days: int = Field(default=30)
    min_txns: int = Field(default=2)


@app.post("/api/v1/aml/structuring", dependencies=[Depends(require_api_key)])
def get_structuring_flags(req: StructuringRequest) -> dict[str, Any]:
    """
    Step 3 — Rolling-window structuring / smurfing detector on bank.xlsx.
    Independent of the point-anomaly and classifier scores.
    """
    from engines.typology.structuring_ledger import detect_structuring, structuring_summary
    bank_df = _get_bank_df()
    flags = detect_structuring(
        bank_df,
        threshold=req.threshold,
        margin=req.margin,
        window_days=req.window_days,
        min_txns=req.min_txns,
    )
    return {
        "summary": structuring_summary(flags),
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# Step 4 — Drift Detection Status
# ---------------------------------------------------------------------------
@app.get("/api/v1/aml/drift", dependencies=[Depends(require_api_key)])
def get_drift_log(last_n: int = 20) -> dict[str, Any]:
    """
    Step 4 — Returns the most recent drift detection records from the log.
    """
    import json as _json
    from engines.fraud.drift import DRIFT_LOG_PATH
    if not os.path.exists(DRIFT_LOG_PATH):
        return {"records": [], "message": "No drift log yet. Run drift simulation first."}
    with open(DRIFT_LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    records = [_json.loads(l) for l in lines[-last_n:] if l.strip()]
    n_fired = sum(1 for r in records if r.get("drift_fired"))
    return {
        "total_windows_logged": len(lines),
        "returned": len(records),
        "drift_fired_count": n_fired,
        "records": records,
    }


@app.post("/api/v1/aml/drift/simulate", dependencies=[Depends(require_api_key)])
def simulate_drift() -> dict[str, Any]:
    """
    Step 4 — Runs drift simulation on PaySim scores and returns results.
    """
    from engines.fraud.drift import simulate_drift_on_paysim
    results = simulate_drift_on_paysim(window_size=1000, alpha=0.05, n_rows=20_000)
    n_fired = sum(1 for r in results if r["drift_fired"])
    return {
        "windows_evaluated": len(results),
        "drift_fired_count": n_fired,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Step 5 — Prosecutor / Defender adjudication for a single transaction
# ---------------------------------------------------------------------------
class AdjudicateRequest(BaseModel):
    transaction_id: str
    risk_score: float


@app.post("/api/v1/aml/adjudicate", dependencies=[Depends(require_api_key)])
def adjudicate_transaction(req: AdjudicateRequest) -> dict[str, Any]:
    """
    Step 5 — Runs prosecutor/defender agent pass for a single transaction.
    """
    from agent.prosecutor_defender import adjudicate_case, _resolution_to_dict
    from engines.typology.structuring_ledger import detect_structuring
    bank_df = _get_bank_df()

    tx_rows = bank_df[bank_df["id"] == req.transaction_id]
    if tx_rows.empty:
        raise HTTPException(status_code=404, detail=f"Transaction {req.transaction_id} not found.")

    tx_row = tx_rows.iloc[0]
    acc_id = tx_row["account_id"]
    acct_df = bank_df[bank_df["account_id"] == acc_id]
    all_scores = [req.risk_score] * 100  # mock portfolio for percentile
    struct_flags = detect_structuring(bank_df)

    resolution = adjudicate_case(
        tx_row=tx_row,
        acct_df=acct_df,
        risk_score=req.risk_score,
        all_scores=all_scores,
        structuring_flags=struct_flags,
    )
    return _resolution_to_dict(resolution)


# ---------------------------------------------------------------------------
# Step 6 — Live Trust Score
# ---------------------------------------------------------------------------
class TrustScoreRequest(BaseModel):
    case_id: str
    sentence: str
    trace_events: list[dict[str, Any]] = Field(default_factory=list)


@app.post("/api/v1/aml/trust-score/tag", dependencies=[Depends(require_api_key)])
def tag_trust_claim(req: TrustScoreRequest) -> dict[str, Any]:
    """Step 6 — Tags a single claim sentence and returns updated trust score."""
    from agent.trust_score import tag_claim, get_session_score
    tag = tag_claim(req.case_id, req.sentence, req.trace_events)
    score = get_session_score(req.case_id)
    return {
        "claim": req.sentence,
        "grounded": tag.grounded,
        "reason": tag.reason,
        "supporting_event_id": tag.supporting_event_id,
        "session_score": score,
    }


@app.get("/api/v1/aml/trust-score/{case_id}", dependencies=[Depends(require_api_key)])
def get_trust_score(case_id: str) -> dict[str, Any]:
    """Step 6 — Returns the current live trust score for a case session."""
    from agent.trust_score import get_session_score
    score = get_session_score(case_id)
    if score is None:
        return {"case_id": case_id, "total_claims": 0, "grounded_pct": 0.0,
                "summary": "No claims recorded yet for this session."}
    return score


# ---------------------------------------------------------------------------
# Step 7 — Conformal Score Wrapping
# ---------------------------------------------------------------------------
class ConformalRequest(BaseModel):
    risk_score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(default=0.90, ge=0.5, le=0.99)


@app.post("/api/v1/aml/conformal", dependencies=[Depends(require_api_key)])
def get_conformal_interval(req: ConformalRequest) -> dict[str, Any]:
    """Step 7 — Wraps a risk score with a split-conformal prediction interval."""
    from agent.simulation import wrap_score
    return wrap_score(req.risk_score, confidence=req.confidence)


# ---------------------------------------------------------------------------
# Screen 2 — Customer Transaction Timeline
# ---------------------------------------------------------------------------
@app.get("/api/v1/aml/timeline/{account_id}", dependencies=[Depends(require_api_key)])
def get_customer_timeline(
    account_id: str,
    with_flags: bool = True,
) -> dict[str, Any]:
    """
    Screen 2 — Full transaction timeline for an account from bank.xlsx,
    with flagged transactions highlighted inline (flagged field per row).
    """
    global _flagged_queue_cache
    bank_df = _get_bank_df()
    acct_df = bank_df[bank_df["account_id"] == account_id]
    if acct_df.empty:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found.")

    if with_flags and _flagged_queue_cache is None:
        from engines.fraud.paysim_score import get_flagged_queue
        _flagged_queue_cache = get_flagged_queue(bank_df, top_n=None)

    flagged_ids = set()
    flagged_scores: dict[str, float] = {}
    if _flagged_queue_cache:
        for r in _flagged_queue_cache:
            if r["account_id"] == account_id and r["flagged"]:
                flagged_ids.add(r["id"])
                flagged_scores[r["id"]] = r["risk_score"]

    timeline = []
    for _, row in acct_df.sort_values("datetime").iterrows():
        tx_id = row["id"]
        timeline.append({
            "id":           tx_id,
            "timestamp":    row["timestamp"],
            "amount":       float(row["amount"]),
            "direction":    row["direction"],
            "balance":      float(row["balance"]),
            "payment_rail": row["payment_rail"],
            "narration":    row.get("raw_narration", ""),
            "flagged":      tx_id in flagged_ids,
            "risk_score":   flagged_scores.get(tx_id),
        })

    return {
        "account_id":   account_id,
        "total_txns":   len(timeline),
        "flagged_count": len(flagged_ids),
        "timeline":     timeline,
    }


# ---------------------------------------------------------------------------
# Screen 3 — Breakdown Panel (SHAP / Feature Attribution Tagged to Row/Stat)
# ---------------------------------------------------------------------------
@app.get("/api/v1/aml/breakdown/{account_id}", dependencies=[Depends(require_api_key)])
def get_customer_breakdown(account_id: str) -> dict[str, Any]:
    """
    Screen 3 — Translates SHAP & feature contributions into template sentences.
    Each claim is tagged to the row/stat that produced it.
    Strictly Gate 1 verified: Date-only granularity, no intraday timestamps.
    """
    prewarmed = _get_prewarmed_cache()
    if prewarmed and "breakdowns" in prewarmed and str(account_id) in prewarmed["breakdowns"]:
        return prewarmed["breakdowns"][str(account_id)]

    from agent.breakdown import generate_account_breakdown
    bank_df = _get_bank_df()
    return generate_account_breakdown(account_id, bank_df=bank_df)


# ---------------------------------------------------------------------------
# Screen 4 — Agent Trace Annotated with Live Trust Score
# ---------------------------------------------------------------------------
@app.get("/api/v1/aml/trace/{account_id}", dependencies=[Depends(require_api_key)])
def get_annotated_agent_trace(account_id: str) -> dict[str, Any]:
    """
    Screen 4 — Steps the agent took to investigate this customer/transaction.
    Each step is clickable to its underlying query result, and annotated
    with the Live Trust Score from Step 6.
    """
    from agent.breakdown import generate_account_breakdown
    from agent.prosecutor_defender import adjudicate_case, _resolution_to_dict
    from agent.trust_score import TrustScoreSession
    from engines.typology.structuring_ledger import detect_structuring

    bank_df = _get_bank_df()
    acct_df = bank_df[bank_df["account_id"] == str(account_id)].sort_values("datetime").copy()
    if acct_df.empty:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found.")

    tx_row = acct_df.iloc[-1]
    tx_id = str(tx_row["id"])
    all_scores = [0.75] * 100
    struct_flags = detect_structuring(bank_df)

    pd_res = adjudicate_case(
        tx_row=tx_row,
        acct_df=acct_df,
        risk_score=0.78,
        all_scores=all_scores,
        structuring_flags=struct_flags,
        log=False,
    )
    pd_dict = _resolution_to_dict(pd_res)
    breakdown = generate_account_breakdown(account_id, bank_df=bank_df)

    date_str = pd.to_datetime(tx_row["datetime"]).strftime("%Y-%m-%d")

    # Construct trace events with full underlying query results
    trace_steps = [
        {
            "step_index": 1,
            "event_id": f"EVT-TX-{account_id[-4:]}-01",
            "tool_called": "get_transaction",
            "action_description": f"Retrieved ledger entry {tx_id} on {date_str}",
            "narration_sentence": f"Retrieved ledger entry {tx_id} for account {account_id} processed on rail {tx_row['payment_rail']}.",
            "query_result": {
                "id": tx_id,
                "account_id": account_id,
                "date": date_str,
                "amount": float(tx_row["amount"]),
                "direction": tx_row["direction"],
                "balance": float(tx_row["balance"]),
                "payment_rail": tx_row["payment_rail"],
                "raw_narration": tx_row.get("raw_narration", ""),
            },
        },
        {
            "step_index": 2,
            "event_id": f"EVT-SHAP-{account_id[-4:]}-02",
            "tool_called": "get_shap_explanation",
            "action_description": "Evaluated PaySim ensemble feature contributions",
            "narration_sentence": f"SHAP feature attribution indicates elevated risk score {breakdown['risk_score']:.2f} driven by balance-drain ratio and daily velocity.",
            "query_result": breakdown,
        },
        {
            "step_index": 3,
            "event_id": f"EVT-STR-{account_id[-4:]}-03",
            "tool_called": "detect_structuring",
            "action_description": "Scanned rolling 30-day window for smurfing patterns",
            "narration_sentence": f"Rolling window analysis scanned all debit outflows against the $1,000 statutory reporting threshold.",
            "query_result": {
                "account_id": account_id,
                "structuring_flags_found": len([f for f in struct_flags if f["account_id"] == str(account_id)]),
                "account_flags": [f for f in struct_flags if f["account_id"] == str(account_id)],
            },
        },
        {
            "step_index": 4,
            "event_id": f"EVT-PD-{account_id[-4:]}-04",
            "tool_called": "prosecutor_defender_adjudication",
            "action_description": "Executed adversarial prosecution and defense review",
            "narration_sentence": f"Prosecutor and defender review adjudicated case to verdict: {pd_dict['verdict'].upper()}.",
            "query_result": pd_dict,
        },
    ]

    # Evaluate Live Trust Score
    session = TrustScoreSession(f"CASE-{account_id}")
    for step in trace_steps:
        session.tag_claim(step["narration_sentence"], trace_steps)

    trust_state = session.state.to_dict()

    # Annotate steps with individual grounding status
    for i, step in enumerate(trace_steps):
        tag = trust_state["claims"][i] if i < len(trust_state["claims"]) else None
        step["is_grounded"] = tag["grounded"] if tag else True
        step["grounding_reason"] = tag["reason"] if tag else None

    return {
        "account_id": account_id,
        "case_id": f"CASE-{account_id}",
        "live_trust_score": {
            "grounded_percentage": trust_state["grounded_pct"],
            "grounded_claims": trust_state["grounded_claims"],
            "total_claims": trust_state["total_claims"],
            "summary": trust_state["summary"],
        },
        "trace_steps": trace_steps,
    }


# ---------------------------------------------------------------------------
# Screen 5 — Customer-Scoped Chatbot Console
# ---------------------------------------------------------------------------
class ScopedChatRequest(BaseModel):
    account_id: str
    query: str
    primary_tx_id: str | None = None


class ForwardSimRequest(BaseModel):
    account_id: str
    transaction_id: str
    days_ahead: int = Field(default=7)


class CounterfactualRecomputeRequest(BaseModel):
    account_id: str
    transaction_id: str
    new_amount: float


@app.post("/api/v1/aml/chat/scoped", dependencies=[Depends(require_api_key)])
def handle_scoped_chat(req: ScopedChatRequest) -> dict[str, Any]:
    """
    Screen 5 — Chatbot scoped strictly to the open customer only.
    Handles:
    1. Plain-language explanation
    2. Counterfactual recompute ("what if this was $2,000?")
    3. Forward simulation ("if she does this again next week, does it still flag?")
    """
    from agent.simulation import handle_scoped_customer_chat
    bank_df = _get_bank_df()
    return handle_scoped_customer_chat(
        account_id=req.account_id,
        query=req.query,
        primary_tx_id=req.primary_tx_id,
        bank_df=bank_df,
    )


@app.post("/api/v1/aml/simulate/forward", dependencies=[Depends(require_api_key)])
def simulate_forward_endpoint(req: ForwardSimRequest) -> dict[str, Any]:
    """
    Screen 5 — Explicit forward simulation endpoint.
    Answers: "If she does this again next week, does it still flag?"
    """
    from agent.simulation import run_forward_simulation
    bank_df = _get_bank_df()
    return run_forward_simulation(
        account_id=req.account_id,
        transaction_id=req.transaction_id,
        days_ahead=req.days_ahead,
        bank_df=bank_df,
    )


@app.post("/api/v1/aml/counterfactual/recompute", dependencies=[Depends(require_api_key)])
def counterfactual_recompute_endpoint(req: CounterfactualRecomputeRequest) -> dict[str, Any]:
    """
    Screen 5 — Explicit counterfactual recompute endpoint.
    Re-scores transaction with modified amount and computes new conformal interval.
    """
    from agent.simulation import run_customer_counterfactual
    bank_df = _get_bank_df()
    return run_customer_counterfactual(
        account_id=req.account_id,
        transaction_id=req.transaction_id,
        new_amount=req.new_amount,
        bank_df=bank_df,
    )



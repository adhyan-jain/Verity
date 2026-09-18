# Verity — Person C Playbook & Implementation Blueprint
**Role:** Agent Core Architect & Grounded Reasoning Engineer  
**Assigned Modules:** `agent/tools.py`, `agent/loop.py`, `agent/grounding.py`, `agent/fallback.py`, `agent/api.py`, `agent/llm.py`, `agent/model_engine.py`, `tests/`  
**Git Branch:** `feature/agent-core`  
**Version:** 2.0.0 (Production Blueprint with Real LLM Loop, Strict Grounding & Model-Backed Inference)  

---

## 1. Executive Context: What is Verity & Why Does It Exist?

### 1.1 The Persona: Priya's Single Desk
Verity is built for **Priya**, a financial-crime analyst at a mid-size digital-first bank / NBFC.
* **The Reality:** Unlike Tier-1 global institutions with separated silos, Priya's desk handles **both** transactional card fraud (high-velocity, point-in-time alerts) and AML account risk (running balance breaks, velocity spikes, and network laundering).
* **The Pain Point:** Swivel-chairing between disjointed fraud dashboards and AML batch reports, dealing with alert fatigue, and suffering from opaque ML models or dangerous, hallucinated LLM summaries that cannot stand up in a regulatory compliance audit.
* **Verity's Promise:** A unified investigation cockpit powered by two specialized detection engines and **one transparent, strictly grounded AI agent** that Priya can interrogate live.

### 1.2 The 4-Person Division of Labor
| Person | Role | Domain | Primary Artifacts |
|---|---|---|---|
| **Person A** | Fraud Engine | Card fraud modeling, SMOTE vs Class Weighting, TreeSHAP explainers | `engines/fraud/*` (Port 8001) |
| **Person B** | Ledger & Typology | 10-account real ledger baseline reconciler + FATF synthetic network generator & graph walker | `engines/ledger/*` (Port 8002)<br>`engines/typology/*` (Port 8003) |
| **Person C (YOU)** | **Agent Core** | **Strictly typed tools, dynamic LLM loop, strict evidence grounding filter, model-backed counterfactuals, live SSE trace streaming, agent API** | `agent/*` (Port 8000)<br>`tests/*` |
| **Person D** | Dashboard UI | Analyst cockpit, case queue, real ledger timeline strip, synthetic typology graph, live trace panel, interactive chat console | `dashboard/*` |

---

## 2. Person C: Role Definition, Boundaries & Core Invariants

As **Person C**, you are the **linchpin of integrity** in Verity. You build the reasoning engine that translates raw mathematical outputs (SHAP scores, balance break anomalies, graph walk steps) into auditable case narratives.

### 2.1 What Exactly You Own
You have exclusive write authority over:
1. `agent/tools.py`: The quarantine layer. The **only** way any LLM or agent logic touches data. Exposes exactly 4 functions.
2. `agent/llm.py`: The hand-rolled LLM client with structured tool selection and conversational evidence-grounded chat.
3. `agent/loop.py`: The dynamic multi-step investigative loop (Case $\to$ LLM $\to$ Tool $\to$ Trace Event $\to$ LLM $\to$ Finish).
4. `agent/model_engine.py`: The calibrated mathematical logistic regression engine for model-backed counterfactuals.
5. `agent/grounding.py`: The strict evidence-only sentence filter. Zero token-overlap loopholes; eliminates unsupported offshore/cartel claims at code level.
6. `agent/fallback.py`: The latency watchdog and cached benchmark Q&A engine for fast, honest demo responses.
7. `agent/api.py`: The FastAPI server exposing real-time SSE trace streaming, case investigations, counterfactuals, and grounded AI chat.
8. `tests/test_agent_*.py`: Automated verification suites proving 100% contract compliance and grounding integrity.

### 2.2 The 5 Inviolable Laws of Person C

> [!IMPORTANT]
> **Law 1: Zero Agent Frameworks (No LangChain, AutoGen, or CrewAI)**  
> The agent loop MUST be an explicit, hand-rolled Python loop. Black-box frameworks add uncontrollable prompt bloat, non-deterministic latency spikes, and opaque failure states. Every step must be transparent and auditable.

> [!IMPORTANT]
> **Law 2: Strict Evidence-Only Grounding (No Lexical Overlap Loopholes)**  
> Telling an LLM "do not hallucinate" fails under stress. In Verity, `agent/grounding.py` programmatically decomposes generated candidate narratives into sentences and verifies that every factual entity, amount, and predicate links directly to a verified `AgentTraceEvent`. Any sentence containing novel speculative claims (e.g. "offshore account", "Cayman Islands", "unregistered cartel") without trace evidence is **strictly purged at code level**.

> [!IMPORTANT]
> **Law 3: Tool Quarantine (Exactly 4 Tools)**  
> The agent interacts with the world strictly through:
> 1. `get_transaction(transaction_id: str)`
> 2. `get_shap_explanation(transaction_id: str)`
> 3. `walk_graph(account_id: str, tier: str, depth: int)`
> 4. `counterfactual(transaction_id: str, parameter_overrides: dict)`  
> The agent is never granted raw SQL/ORM access, shell access, or unbounded API freedom.

> [!IMPORTANT]
> **Law 4: True Model-Backed Counterfactuals**  
> When an analyst asks "What if the amount was $50 instead of $4,850?", the agent does not guess or use hardcoded step thresholds. It modifies the 30-feature vector and re-runs inference through the calibrated mathematical scoring model (`agent/model_engine.py`) to compute the exact probability shift and feature attribution delta.

> [!IMPORTANT]
> **Law 5: Zero-Blocking Dual-Mode Architecture & Real-Time SSE Streaming**  
> Never wait for Person A to finish training models or Person B to finish parsing Excel sheets. All tools support dual-mode execution (`VERITY_ENV=mock` vs `VERITY_ENV=live`). Live investigation events stream to Person D's dashboard via Server-Sent Events (`text/event-stream`).

---

## 3. Data Contracts Reference (Person C's Schema Bible)

All Person C modules must strictly consume and produce data conforming to `contracts/schemas.json`.

```
┌────────────────────────────────────────────────────────┐
│                   AgentTraceEvent                      │
├────────────────────────────────────────────────────────┤
│ event_id: string (e.g. "EVT-101")                      │
│ case_id: string (e.g. "CASE-CARD-001")                 │
│ timestamp: ISO8601 (e.g. "2026-09-18T09:15:00Z")       │
│ tool_called: enum ["get_transaction",                  │
│                    "get_shap_explanation",             │
│                    "walk_graph",                       │
│                    "counterfactual"]                   │
│ tool_input: object                                     │
│ tool_output_summary: string                            │
│ narration_sentence: string                             │
└────────────────────────────────────────────────────────┘
                           │
                           ▼ (Passes through agent/grounding.py)
┌────────────────────────────────────────────────────────┐
│                         Case                           │
├────────────────────────────────────────────────────────┤
│ case_id: string                                        │
│ tier_origin: "real_card" | "real_ledger" |             │
│              "synthetic_network"                       │
│ status: "open" | "investigating" | "closed"            │
│ primary_transaction_id: string                         │
│ risk_score: number (0.0 to 1.0)                        │
│ trace_events: Array<AgentTraceEvent>                   │
│ narrative: string (Concatenation of verified sentences)│
└────────────────────────────────────────────────────────┘
```

---

## 4. Dynamic LLM Loop & Strict Grounding Architecture

```mermaid
sequenceDiagram
    autonumber
    participant UI as Analyst Dashboard
    participant API as agent/api.py (SSE)
    participant Loop as agent/loop.py
    participant LLM as agent/llm.py
    participant Tool as agent/tools.py
    participant Model as agent/model_engine.py
    participant Ground as agent/grounding.py

    UI->>API: POST /api/v1/agent/investigate
    API->>Loop: run_investigation_loop()
    
    loop Dynamic Tool Selection (up to 4 steps)
        Loop->>LLM: decide_next_step(case, history, step)
        LLM-->>Loop: {thought, action, action_input}
        Loop->>Tool: execute(action, action_input)
        Tool-->>Loop: Raw Output / Engine Evidence
        Loop->>Loop: Create AgentTraceEvent
        Loop->>API: on_step_callback(event)
        API-->>UI: SSE Push: data: AgentTraceEvent
    end

    Loop->>LLM: Request Final Candidate Narrative
    LLM-->>Loop: candidate_narrative
    Loop->>Ground: ground_narrative(trace_events, candidate_narrative)
    Ground->>Ground: Strict Evidence Validation (prune unsupported claims)
    Ground-->>Loop: grounded_narrative
    Loop-->>API: Grounded Case Object
    API-->>UI: HTTP 200 Final Case
```

---

## 5. Verification Checklist & Test Suites

The test suite in `tests/` contains **33 automated tests** verifying every functional requirement:

```powershell
pytest -v
```

1. **`tests/test_agent_tools.py` (8 tests):**
   - Verified `get_transaction`, `get_shap_explanation`, `walk_graph` (real ledger vs synthetic network).
   - Verified `counterfactual`: model-backed probability reduction on Amount decrease with feature attribution deltas.
2. **`tests/test_agent_grounding.py` (6 tests):**
   - Verified preservation of currency decimals and timestamps.
   - Verified **The Critique's Offshore Hallucination Test**: Candidate sentence claiming an "offshore account" is strictly purged despite high token overlap.
   - Verified rejection of unsupported entity IDs (`ACC-9999`) and fabricated numbers (`$99,000.00`).
   - Verified `audit_grounding()` tracking pruned sentences and reasons.
3. **`tests/test_agent_loop.py` (5 tests):**
   - Verified dynamic LLM decision-making (`test_llm_decision_step`).
   - Verified real ledger balance degradation calculation (-$3,200.00 overdraft).
   - Verified synthetic network cycle verification and 96.9% volume retention calculation.
   - Verified real-time streaming callback.
4. **`tests/test_agent_fallback.py` (6 tests):**
   - Verified pattern matching for all 4 core judge queries.
   - Verified latency guard watchdog triggering fallback notice upon timeout.
5. **`tests/test_agent_api.py` (8 tests):**
   - Verified `/health` probe (version 2.0.0).
   - Verified `/investigate` case endpoint.
   - Verified `/trace-stream` JSON polling and **SSE streaming (`text/event-stream`)**.
   - Verified `/counterfactual` model-backed calculation.
   - Verified `/chat` conversational AI query and latency watchdog fallback.

---

## 6. Integration Milestones & Pitch Defense

### Review 1 (5:00 PM Milestone)
- Person C delivers: Model-backed card fraud investigation with SHAP interpretable/anonymized split.
- Person D displays: Unified case queue and SHAP factor attribution visualizer.

### Review 2 (1:00 AM Milestone)
- Person C delivers: Multi-tier dynamic loop, live SSE trace streaming, and interactive chat console with latency watchdog.
- Person D displays: Real ledger horizontal timeline and synthetic FATF typology graph.

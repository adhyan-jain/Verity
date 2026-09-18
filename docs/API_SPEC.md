# API Specification & Data Contracts — Verity

This document outlines the REST API endpoints and data contracts actually exposed by Verity's
four services, verified against the implementation (`engines/*/api.py`, `agent/api.py`) and
against `scripts/smoke_test.py`, which exercises every endpoint below over real HTTP against
real data. All payload shapes conform to `contracts/schemas.json`.

Config lives in one place: `.env` (copy from `.env.example` at the repo root). See
`README.md` → "Run" for the one-command bring-up.

---

## 1. Fraud Engine API (`engines/fraud/api.py`)

**Base URL:** `http://localhost:8001` (routes also mounted under `/api/v1/fraud`)

**Auth:** every route below except `/health` requires header `X-API-Key: <FRAUD_API_KEY>`.
The service fails closed — if `FRAUD_API_KEY` is unset server-side, every protected route
returns `503`; a wrong/missing key returns `401`. `agent/tools.py` forwards this header
automatically in live mode; the dashboard forwards it from `VITE_FRAUD_API_KEY` (see
ARCHITECTURE.md for why that's a prototype-only shortcut).

### 1.1 Health
* `GET /health` / `GET /api/v1/fraud/health` — unauthenticated. Returns model version, training
  strategy, and held-out metrics. `503` if the model artifact failed to load at startup.

### 1.2 Get Transaction Details
* `GET /api/v1/fraud/transaction/{transaction_id}` — `transaction_id` may be `TX-CARD-<n>` or a
  bare row index; `<n>` is parsed as the row index into `data/raw/creditcard.csv`.
```json
{
  "id": "TX-CARD-9842",
  "tier": "real_card",
  "timestamp": "2026-09-18T03:22:00Z",
  "account_id": null,
  "amount": 4850.00,
  "direction": "debit",
  "raw_narration": null,
  "source_dataset": "creditcard.csv"
}
```
`404` if the index is out of range; `400` if no digits can be parsed from the ID.

### 1.3 Get SHAP Feature Attribution
* `GET /api/v1/fraud/explain/{transaction_id}` — same ID rule as above.
```json
{
  "transaction_id": "TX-CARD-9842",
  "risk_score": 0.89,
  "verdict": "flagged",
  "top_factors": [
    { "feature": "Amount", "human_label": "Transaction amount ($4,850.00)", "contribution": 0.42, "interpretable": true },
    { "feature": "Time", "human_label": "Transaction time (03:22 AM)", "contribution": 0.19, "interpretable": true },
    { "feature": "V14", "human_label": "Anonymized behavioral signal V14", "contribution": 0.28, "interpretable": false }
  ],
  "model_version": "v1.2-..."
}
```

### 1.4 Counterfactual
* `POST /api/v1/fraud/counterfactual` (also mounted at `POST /api/v1/agent/counterfactual` on
  this same service — a duplicate route registered directly on the fraud app; the one actually
  used end-to-end is the agent's own `/api/v1/agent/counterfactual`, §4.3 below).
```json
// Request
{ "transaction_id": "TX-CARD-9842", "parameter_overrides": { "Amount": 150.00, "Time": 50000 } }
// Response
{
  "transaction_id": "TX-CARD-9842",
  "original_risk_score": 0.89, "recalculated_risk_score": 0.18,
  "original_verdict": "flagged", "recalculated_verdict": "clear",
  "modifications": { "Amount": 150.00, "Time": 50000 }
}
```

### 1.5 List Transactions
* `GET /api/v1/fraud/transactions?limit=&offset=&flagged_only=` — paginated listing with
  ground-truth `Class` labels for demo/QA use. Not documented in earlier drafts; exists in code.

---

## 2. Ledger Engine API (`engines/ledger/api.py`)

**Base URL:** `http://localhost:8002/api/v1/ledger` · **Auth:** none · **CORS:** `*`

### 2.1 Health
* `GET /health`

### 2.2 List Accounts
* `GET /accounts` — summaries for all real accounts parsed from `data/raw/bank.xlsx`. Real
  account IDs are literal bank account numbers (e.g. `"409000362497"`), **not** the
  `ACC-1092`-style IDs used in `contracts/mock_data/` — see ARCHITECTURE.md "Remaining gaps".

### 2.3 Reconciliation Anomalies
* `GET /anomalies` — all detected anomalies across every account.
* `GET /anomalies/{account_id}` — anomalies for one account.
```json
[{
  "account_id": "409000362497", "anomaly_type": "timing_spike",
  "window_start": "2026-09-15T00:00:00Z", "window_end": "2026-09-17T23:59:59Z",
  "severity": 0.88, "baseline_value": 2.1, "observed_value": 9.4,
  "evidence_transaction_ids": ["TX-LEDGER-003011", "TX-LEDGER-003012"]
}]
```

### 2.4 Timeline
* `GET /timeline/{account_id}` — chronological transactions + overlaid anomaly windows.
```json
{
  "account_id": "409000362497", "account_name": "...", "total_transactions": 48,
  "anomalies": [{ "anomaly_type": "balance_break", "severity": 0.92, "window_start": "...", "window_end": "...", "baseline_value": 15400.0, "observed_value": -3200.0 }],
  "transactions": [{ "id": "TX-LEDGER-003011", "timestamp": "...", "amount": 12500.0, "direction": "debit", "balance": -800.0, "narration": "..." }]
}
```
`404` if the account doesn't exist.

### 2.5 Walk (single-account sequential history)
* `GET /walk/{account_id}?limit=20` — real-tier "walk" is the account's own transaction
  sequence (no cross-account claims — see `VERITY_BUILD_SPEC.md` §3). Returns `GraphWalkStep[]`.

### 2.6 Get Transaction
* `GET /transaction/{transaction_id}` — exact match on the parsed `id` column
  (`TX-LEDGER-000001` format, sequential post-sort). `404` if not found.

---

## 3. Typology Engine API (`engines/typology/api.py`)

**Base URL:** `http://localhost:8003/api/v1/typology` · **Auth:** none · **CORS:** `*`

### 3.1 Health
* `GET /health`

### 3.2 Typology Flags
* `GET /flags` — all FATF flags detected across the synthetic network.
```json
[{
  "flag_id": "FLAG-FATF-001", "typology": "round_tripping",
  "fatf_reference": "FATF Guidance on Concealment of Beneficial Ownership (Oct 2018)",
  "involved_accounts": ["ACC-RT-204", "ACC-RT-501", "..."],
  "evidence_transaction_ids": ["TX-SYNTH-0027", "..."], "confidence": 0.94
}]
```
Real generated account IDs are seeded by role: `ACC-SMURF-*`, `ACC-RT-*` (round-trip),
`ACC-LAYER-*`, `ACC-BENIGN-*` — not the generic `ACC-SYN-401`-style IDs used in
`contracts/mock_data/`. Regenerate via `python -m data.synthetic.generate_network` (seed 42,
deterministic).

### 3.3 Full Network
* `GET /network` — `{ metadata, nodes[], edges[], ground_truth_flags[] }` for graph rendering.

### 3.4 Walk (multi-hop BFS)
* `GET /walk/{account_id}?depth=2` — BFS traversal from `account_id` up to `depth` hops.
  Returns `[]` (not an error) if `account_id` has no outbound edges in the live network.

### 3.5 Adversarial Evaluation
* `GET /evaluation` — precision/recall of `fatf_rules.py`'s detectors against
  `data/synthetic/adversarial_set.json` (held-out set, built independently of `detect.py`'s
  author per `VERITY_BUILD_SPEC.md`).

---

## 4. Agent Core API (`agent/api.py`)

**Base URL:** `http://localhost:8000` · **Auth:** none · **CORS:** `*`

### 4.1 Health
* `GET /health`

### 4.2 Investigate
* `POST /api/v1/agent/investigate`
```json
// Request
{ "case_id": "CASE-CARD-001", "transaction_id": "TX-CARD-9842", "tier_origin": "real_card", "simulate_latency": 0 }
```
Runs the dynamic tool-calling loop (up to 4 steps): `get_transaction` → tier-specific evidence
(`get_shap_explanation` for `real_card`, `walk_graph` for `real_ledger`/`synthetic_network`) →
grounded narrative. Returns a `Case` object (`contracts/schemas.json`). Wrapped in a 20s latency
watchdog (`agent/fallback.py`) — on timeout or unhandled exception it returns a cached benchmark
answer instead, with `fallback_notice` set (never a silent failure).

### 4.3 Counterfactual
* `POST /api/v1/agent/counterfactual` — `{ "transaction_id": "...", "parameter_overrides": {...} }`.
  Re-runs real model inference (`agent/model_engine.py` in mock mode, or the fraud engine's own
  `/counterfactual` in live mode) — never a hardcoded threshold or an LLM guess.

### 4.4 Chat
* `POST /api/v1/agent/chat` — `{ "case_id": "...", "query": "...", "simulate_latency": 0 }`.
  Checks `contracts/mock_data/mock_fallback_qa.json` for the 4 benchmark questions first; falls
  through to conversational LLM reasoning over the case's trace events, then the grounding
  filter (`agent/grounding.py`) before returning.

### 4.5 Trace Stream
* `GET /api/v1/agent/trace-stream/{case_id}` — JSON array of `AgentTraceEvent`s by default;
  `?stream=true` or `Accept: text/event-stream` switches to a live SSE stream.

---

## 5. Agent Tools Contract (`agent/tools.py`)

Exactly four functions are callable by the LLM loop — no other data access exists:

```python
def get_transaction(transaction_id: str) -> TransactionRecord: ...
def get_shap_explanation(transaction_id: str) -> FraudExplanation: ...
def walk_graph(account_id: str, tier: str = "real_ledger", depth: int = 2) -> dict: ...  # {account_id, tier, steps: GraphWalkStep[]}
def counterfactual(transaction_id: str, parameter_overrides: dict) -> CounterfactualResult: ...
```

`VERITY_ENV=live` routes each call to the engine that actually owns the ID
(`get_transaction`: `TX-LEDGER-*` → ledger, `TX-SYNTH-*`/`TX-SYN-*` → typology's network edges,
everything else → fraud), falling back to `contracts/mock_data/` fixtures on any timeout,
connection error, or unmatched live lookup. `VERITY_ENV=mock` (default) skips HTTP entirely.

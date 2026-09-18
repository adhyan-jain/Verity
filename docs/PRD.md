# Product Requirements Document (PRD) — Verity

**Product Name:** Verity  
**Document Version:** 1.0.0  
**Target Audience:** Hackathon Judges, Engineering Team, Financial Crime Analysts  
**Document Owner:** Verity Core Team  

---

## 1. Executive Summary & Vision

### 1.1 One-Liner
**Verity** is a unified financial-crime analyst workspace — two detection engines (card fraud and account-level ledger risk) feeding a single case queue, paired with an AI agent that investigates each flagged case and can be questioned about its own reasoning live.

### 1.2 Target Persona
* **Name:** Priya
* **Role:** Financial-Crime Analyst
* **Organization:** Mid-size digital-first bank / Non-Banking Financial Company (NBFC).
* **Pain Point:** The institution is too lean to run separate card-fraud and AML (Anti-Money Laundering) investigation teams. Priya handles both high-velocity credit card alert triage and deep ledger account-risk investigations. She suffers from alert fatigue, siloed tools, ungrounded AI summaries with hallucinated graph hops, and opaque model predictions.
* **Verity's Promise:** A single, trustworthy cockpit that automates trace-grounded investigations, exposes mathematical SHAP attributions, visualizes sparse ledger timelines alongside complex laundering networks, and answers live counterfactual questions with recomputed models.

---

## 2. Problem Statement & Market Context

1. **Siloed Detection Systems:** Legacy institutions run disparate tooling for transactional card fraud (real-time, point-in-time) and AML transaction monitoring (batch, network-based), forcing analysts to swivel-chair across platforms.
2. **AI Hallucinations in Compliance:** Generic LLMs often hallucinate intermediate transactional hops or assert counterfactual outcomes without backing data, making AI summaries dangerous in auditable compliance workflows.
3. **Data Asymmetry:** Real ledger data for single accounts is often sparse (e.g. 10 accounts), whereas synthetic networks are hyper-connected. Treating them uniformly creates misleading visual representations.
4. **Black-Box Features:** PCA-transformed variables (V1–V28) are frequently misattributed by LLMs attempting to invent human meaning rather than respecting mathematical privacy bounds.

---

## 3. Product Goals & Core Objectives

1. **Dual-Engine Case Queue:** Unify card-fraud scoring and ledger anomaly detection into a single prioritized queue.
2. **Deterministic Code-Level Grounding:** Enforce that 100% of generated narrative sentences are mapped directly to verified `AgentTraceEvent` IDs.
3. **Transparent Explainability:** Clearly partition SHAP feature attributions into human-interpretable factors (Time, Amount) versus mathematically anonymized signals (V1–V28).
4. **Dual-Surface Visualization:** Provide an interactive horizontal transaction timeline for real ledger accounts and a multi-hop graph visualization for synthetic FATF laundering typologies.
5. **Live Counterfactual Investigation:** Enable analysts to ask "what if" queries (e.g., amount changes) with real engine re-evaluations rather than LLM guesses.
6. **Graceful Latency Fallbacks:** Maintain responsiveness via live trace stream updates and transparent pre-computed benchmarks for heavy operations.

---

## 4. Functional Requirements

### 4.1 Detection Engines

#### Engine 1: Card Fraud Detection (`engines/fraud/`)
* **FR-1.1:** Must ingest `data/raw/creditcard.csv` (284,807 transactions).
* **FR-1.2:** Must benchmark imbalance management techniques (SMOTE vs. Class Weighting) on recall and PR-AUC.
* **FR-1.3:** Must expose a trained model (`model.pkl`) outputting `risk_score` (0.0–1.0) and `verdict` (`flagged` vs `clear`).
* **FR-1.4:** Must compute SHAP values and segregate interpretable features (`Time`, `Amount`) from anonymized features (`V1`–`V28`).
* **FR-1.5:** Must expose FastAPI endpoints `GET /api/v1/fraud/transaction/{id}` and `GET /api/v1/fraud/explain/{id}`.

#### Engine 2: Account Ledger Engine (`engines/ledger/`)
* **FR-2.1:** Must ingest and parse narrations from `data/raw/bank.xlsx` across 10 accounts.
* **FR-2.2:** Must compute per-account baseline velocity, running balance, and reversal ratios.
* **FR-2.3:** Must detect three distinct anomaly types:
  * `balance_break`: Unexpected balance drops or sudden overdraft transitions.
  * `timing_spike`: Burst of transaction count/velocity > 3x the 30-day baseline.
  * `reversal_outlier`: Abnormal ratio of debit/credit reversal patterns.
* **FR-2.4:** Must generate timeline datasets mapping transactions and anomaly windows for horizontal strip rendering.
* **FR-2.5:** Must expose FastAPI endpoints `GET /api/v1/ledger/anomalies/{account_id}` and `GET /api/v1/ledger/timeline/{account_id}`.

#### Engine 3: FATF Typology Engine (`engines/typology/` & `data/synthetic/`)
* **FR-3.1:** Must generate synthetic multi-party networks seeded with three FATF laundering typologies:
  * **Structuring (Smurfing):** Multiple sub-threshold transfers within 24 hours.
  * **Round-Tripping:** Circular flows of funds returning to the originating entity.
  * **Rapid Layering:** High-velocity pass-through transfers across $\ge 3$ intermediary hops.
* **FR-3.2:** Must provide graph walk endpoints returning `GraphWalkStep` records.
* **FR-3.3:** Must maintain an isolated hold-out test set (`adversarial_set.json`) to validate typology detection accuracy.

---

### 4.2 AI Grounding & Investigation Agent (`agent/`)

* **FR-4.1 (Tool Interface):** The LLM must interact with data *exclusively* via four strictly typed tools:
  1. `get_transaction(transaction_id)`
  2. `get_shap_explanation(transaction_id)`
  3. `walk_graph(account_id, tier, depth)`
  4. `counterfactual(transaction_id, parameter_overrides)`
* **FR-4.2 (Investigation Loop):** Execute a deterministic step-by-step tool-calling loop that emits an `AgentTraceEvent` for every action.
* **FR-4.3 (Code-Level Grounding Filter):** Enforce via `grounding.py` that the final `narrative` string consists solely of sentences directly derived from generated `AgentTraceEvent`s. Any hallucinated statements must be stripped programmatically before transmission to the frontend.
* **FR-4.4 (Live Status & Fallback):**
  * Stream trace events live to the UI to convey active work.
  * For queries exceeding a 20-second timeout, gracefully serve pre-cached benchmark answers with explicit UI disclosure (`"Using a prepared benchmark answer for this query"`).

---

### 4.3 Dashboard User Interface (`dashboard/`)

* **FR-5.1 (Case Queue):** Prioritized list of alerts combining card fraud cases and ledger anomaly alerts, filterable by risk score, origin tier, and status.
* **FR-5.2 (Explainability Panel):** Visual breakdown of risk factors displaying interpretable metrics alongside anonymized behavioral vectors.
* **FR-5.3 (Tabbed Dual-Surface Visualizer):**
  * *Tab 1 (Real Ledger):* Horizontal timeline visualization highlighting anomaly windows across individual account histories.
  * *Tab 2 (Synthetic Network):* Interactive graph visualization showing multi-hop nodes, edges, and highlighted FATF typology cycles.
* **FR-5.4 (Reasoning Trace Panel):** Real-time chronological audit trail of agent steps and tool outputs.
* **FR-5.5 (Interactive Analyst Chat):** Live query console supporting natural language questioning and parameter counterfactuals.

---

## 5. Non-Functional Requirements (NFRs)

| Metric | Target | Verification Method |
|---|---|---|
| **Grounding Integrity** | 100% sentence citation | Automated unit tests in `agent/grounding.py` verifying zero ungrounded sentences |
| **API Response Time** | < 250ms for cached/indexed queries | FastAPI performance profiling |
| **Agent Step Latency** | Live trace emitted < 2s per hop | Client websocket / polling stream validation |
| **Imbalance Recall** | > 85% recall on fraud class | Imbalance benchmark test harness on test split |
| **Data Privacy / Isolation**| Zero raw data leaks | Raw CSV/XLSX gitignored; strictly structured JSON output |

---

## 6. What We Do NOT Claim (Anti-Scope)

1. **No Real-Ledger Laundering Networks:** We do not claim the 10 real bank accounts form an interconnected international cartel; they are analyzed as independent accounts.
2. **No Reverse-Engineering of PCA:** We do not attempt to guess what V1–V28 mean in human terms; we explain them strictly as mathematical behavioral dimensions.
3. **Not a Production Audited AML Suite:** Verity is an analyst decision-support prototype designed for auditability and rigor.

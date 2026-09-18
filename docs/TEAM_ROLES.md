# Verity — Team Roles, Boundaries & Git Workflow

This document defines the strict division of labor and integration checkpoints so all 4 team members can build in parallel without merge conflicts or blocking dependencies.

---

## 👥 Role Assignments & File Boundaries

| Role | Domain | Assigned Files / Modules | Input / Output Contract | Git Branch |
|---|---|---|---|---|
| **Person A** | **Fraud Engine** | `engines/fraud/benchmark_imbalance.py`<br>`engines/fraud/train.py`<br>`engines/fraud/explain.py`<br>`engines/fraud/api.py` | Consumes `data/raw/creditcard.csv`<br>Produces `TransactionRecord`, `FraudExplanation` | `feature/fraud-engine` |
| **Person B** | **Ledger + Synthetic** | `engines/ledger/parse_narrations.py`<br>`engines/ledger/reconcile.py`<br>`engines/ledger/anomalies.py`<br>`engines/ledger/timeline.py`<br>`engines/ledger/api.py`<br>`data/synthetic/generate_network.py`<br>`engines/typology/fatf_rules.py`<br>`engines/typology/detect.py`<br>`engines/typology/api.py`<br>`data/synthetic/adversarial_set.json` | Consumes `data/raw/bank.xlsx`<br>Produces `ReconciliationAnomaly`, `GraphWalkStep`, `TypologyFlag` | `feature/ledger-typology` |
| **Person C** | **Agent Core** | `agent/tools.py`<br>`agent/loop.py`<br>`agent/grounding.py`<br>`agent/fallback.py` | Dispatches tools, enforces grounding filter, produces `AgentTraceEvent` & `Case` narrative | `feature/agent-core` |
| **Person D** | **Dashboard UI** | `dashboard/case_list/`<br>`dashboard/graph_view/`<br>`dashboard/reasoning_trace_panel/`<br>`dashboard/chat_panel/` | Consumes `Case`, `AgentTraceEvent`, `ReconciliationAnomaly`, `GraphWalkStep` | `feature/dashboard-ui` |

---

## 🔀 Git Branching & Workflow Rules

1. **Never commit directly to `master` / `main`** during development sprints.
2. **Branch Naming**:
   - `git checkout -b feature/fraud-engine`
   - `git checkout -b feature/ledger-typology`
   - `git checkout -b feature/agent-core`
   - `git checkout -b feature/dashboard-ui`
3. **Mock-Data First (Zero Blocking)**:
   - Person C (Agent) and Person D (Dashboard) must develop and test their logic against fixtures in [`contracts/mock_data/`](../contracts/mock_data/) until the first integration checkpoint.
   - Do not wait for models to finish training or Excel sheets to parse.
4. **Integration Protocol**:
   - Merge back into `master` only at the scheduled review milestones (5 PM Review 1, 1 AM Review 2).
   - Resolve contract discrepancies in `contracts/schemas.json` before modifying code.

---

## ⏰ Milestone Schedule

- **Hour 1 (Done)**: Finalize `contracts/schemas.json`, `.gitignore`, `contracts/mock_data/`, repository scaffolding.
- **Review 1 (5:00 PM)**:
  - Demo: Real card fraud case + SHAP explanation + Agent narrating fraud case end-to-end + Ledger timeline view.
- **Review 2 (1:00 AM)**:
  - Demo: Agent traversing real ledger & synthetic network graph + Live trace panel + Interactive chat Q&A / counterfactuals.
- **Freeze & Polish (4:00 AM - 10:00 AM)**:
  - Metrics slide, chat fallback state, prompt tuning, 3+ full rehearsal runs.

# Verity — Financial Crime Analyst Workspace

> **One-Liner:** Verity is a unified financial-crime analyst workspace — two detection engines (card-fraud alerts and ledger account-risk) feeding one case queue, with an AI agent that investigates each flagged case and can be questioned about its own reasoning live.

## 🎯 Target Persona
**Priya**, a financial-crime analyst at a mid-size digital-first bank / NBFC where one combined desk manages both transaction fraud alerts and account-level ledger investigations.

---

## 🏗 Repository Structure

```
Verity/
  data/
    raw/                       # Raw datasets (creditcard.csv, bank.xlsx) - gitignored
    synthetic/                 # Synthetic FATF typology network & adversarial sets
  engines/
    fraud/                     # Imbalance benchmarking, training, SHAP explainability, API
    ledger/                    # Transaction parsing, baselines, anomaly detection, timelines
    typology/                  # FATF rules (structuring, round-tripping, rapid-layering), detection
  agent/
    tools.py                   # Data access layer for agent
    loop.py                    # Hand-rolled tool-calling loop
    grounding.py               # Code-enforced grounding filter (trace event citations)
    fallback.py                # Pre-cached fallback Q&A & live state handling
  contracts/
    schemas.json               # Canonical data contracts
    mock_data/                 # Fixtures and test data
  dashboard/                   # Visual workspace & analyst frontend
    case_list/                 # Unified triage queue
    graph_view/                # Tabbed Real Ledger timeline / Synthetic network
    reasoning_trace_panel/     # Live AgentTraceEvent stream
    chat_panel/                # Live interactive analyst Q&A & counterfactuals
  docs/
    PITCH.md                   # Pitch talking points & "What We Don't Claim"
    DEMO_SCRIPT.md             # Click-path and demonstration flow
  VERITY_BUILD_SPEC.md         # Full project architecture and build specification
```

---

## 🔒 Core Data Contracts
The data contracts governing all engine outputs, agent trace events, and UI case models are defined in [`contracts/schemas.json`](contracts/schemas.json) per [`VERITY_BUILD_SPEC.md`](VERITY_BUILD_SPEC.md).

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+ / npm (for frontend dashboard)

### Setup
1. Clone the repository
2. Place raw datasets (`creditcard.csv`, `bank.xlsx`) into `data/raw/`
3. Refer to [`VERITY_BUILD_SPEC.md`](VERITY_BUILD_SPEC.md) for engine build order and integration milestones.

# Verity — Financial Crime Analyst Workspace

> **One-Liner:** Verity is a unified financial-crime analyst workspace — two detection engines
> (card-fraud alerts and ledger account-risk) feeding one case queue, with an AI agent that
> investigates each flagged case and can be questioned about its own reasoning live.

> **New to this repo?** [`GETTING_STARTED.md`](GETTING_STARTED.md) walks through every step
> from `git clone` to a running, tested stack.

## 🎯 Target Persona
**Priya**, a financial-crime analyst at a mid-size digital-first bank / NBFC where one combined
desk manages both transaction fraud alerts and account-level ledger investigations.

---

## 🏗 Repository Structure

```
Verity/
  data/
    raw/                       # Raw datasets (creditcard.csv, bank.xlsx) — Git LFS
    synthetic/                 # FATF typology network generator + generated network/adversarial set
  engines/
    fraud/                     # Imbalance benchmarking, training, SHAP explainability, FastAPI :8001
    ledger/                    # bank.xlsx parsing, baselines, anomaly detection, timelines, FastAPI :8002
    typology/                  # FATF rules (structuring, round-tripping, rapid-layering), FastAPI :8003
  agent/                       # Hand-rolled tool-calling loop, grounding filter, fallback Q&A, FastAPI :8000
  contracts/
    schemas.json               # Canonical data contracts (single source of truth)
    mock_data/                 # Fixtures — used when VERITY_ENV=mock, and as live-mode fallback
  dashboard/                   # TanStack Start + React analyst cockpit, Vite dev server :3000
  scripts/
    dev_up.py                  # One-command bring-up for all 4 services (+ dashboard)
    smoke_test.py               # End-to-end check against real data, no mocks
  docs/                        # Architecture, PRD, API spec, pitch, demo script
  tests/                       # pytest suite for engines + agent (56 tests)
  ARCHITECTURE.md              # Integration survey: components, gaps, decisions, remaining work
  .env.example                 # All environment variables in one place
```

---

## 🔒 Core Data Contracts
All engine outputs, agent trace events, and UI case models conform to
[`contracts/schemas.json`](contracts/schemas.json). Full endpoint-by-endpoint reference:
[`docs/API_SPEC.md`](docs/API_SPEC.md). System design and integration decisions:
[`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## 🚀 Setup

### Prerequisites
- Python 3.11+
- Node.js 18+ / npm (dashboard uses npm — see ARCHITECTURE.md for why, not bun)
- `data/raw/creditcard.csv` and `data/raw/bank.xlsx` (pulled via Git LFS on clone)

### Install
```sh
# Python services
python -m venv .venv
.venv/Scripts/activate            # .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

# Dashboard
cd dashboard && npm install && cd ..

# Environment
cp .env.example .env              # edit FRAUD_API_KEY / VITE_FRAUD_API_KEY if you change it
```

---

## ▶️ Run

One command starts all four backend services (agent :8000, fraud :8001, ledger :8002, typology
:8003) — training the fraud model first if `engines/fraud/model.pkl` doesn't exist yet (~90s,
one-time) — then the dashboard dev server on :3000:

```sh
python scripts/dev_up.py
```

Backend services only (e.g. if you're running the dashboard separately, or don't need the UI):
```sh
python scripts/dev_up.py --no-dashboard
```

Open **http://localhost:3000**. The masthead shows live engine health (N/4 online); selecting a
case runs a real agent investigation, fetches live SHAP/timeline/network evidence, and the
counterfactual/chat panel hits the real agent API.

Stop everything with Ctrl+C.

### Manual / individual services (for debugging one engine)
```sh
python -m uvicorn agent.api:app --port 8000
python -m uvicorn engines.fraud.api:app --port 8001
python -m uvicorn engines.ledger.api:app --port 8002
python -m uvicorn engines.typology.api:app --port 8003
npm --prefix dashboard run dev
```

---

## ✅ Test

**Unit/integration test suite** (56 tests, mocked/in-process — no services need to be running):
```sh
pytest -v
```

**End-to-end smoke test** (starts all 4 real services against real data, exercises every
critical path over HTTP — fraud SHAP, ledger timeline, typology network, full agent
investigation for all 3 tiers, model-backed counterfactual, chat with cached fallback — then
tears everything down):
```sh
python scripts/smoke_test.py
```

**Dashboard build/typecheck:**
```sh
cd dashboard && npx tsc --noEmit && npm run build
```

---

## 📄 Further reading

- [`GETTING_STARTED.md`](GETTING_STARTED.md) — full step-by-step from a fresh clone, including
  troubleshooting for first-run quirks (LFS, model training, first-call parse latency).
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — component map, every integration gap found, decisions
  made, and what's still unresolved.
- [`docs/API_SPEC.md`](docs/API_SPEC.md) — full endpoint reference, verified against the code.
- [`VERITY_BUILD_SPEC.md`](VERITY_BUILD_SPEC.md) — original design spec (data contracts,
  detection engine internals, agent grounding rules).
- [`docs/PITCH.md`](docs/PITCH.md), [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) — presentation
  materials.

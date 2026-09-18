# Verity — Financial Crime Analyst Workspace

> **One-Liner:** Verity is a unified financial-crime analyst workspace — two detection engines
> (card-fraud alerts and ledger account-risk) feeding one case queue, with an AI agent that
> investigates each flagged case and can be questioned about its own reasoning live.

> **New to this repo?** [`GETTING_STARTED.md`](GETTING_STARTED.md) walks through every step
> from `git clone` to a running, tested stack.

## 🎯 Target Persona
**Chitrita**, a financial-crime analyst at a mid-size digital-first bank / NBFC where one combined
desk manages both transaction fraud alerts and account-level ledger investigations.

---

## 🏗 Repository Structure

```
Verity/
  data/
    raw/                       # Raw datasets (creditcard.csv, bank.xlsx) — Git LFS
    synthetic/                 # FATF typology network generator + generated network/adversarial set
  engines/
    fraud/                     # Imbalance benchmarking, training, SHAP explainability, split-conformal risk interval, FastAPI :8001
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
  tests/                       # pytest suite for engines + agent (80 tests)
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
:8003) — training the fraud model and calibrating its split-conformal risk interval first if
`engines/fraud/model.pkl` / `engines/fraud/conformal.pkl` don't exist yet (~90s + ~10s,
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

## ✅ Test & Rigor Evaluation

**Unit/integration test suite** (80 tests, mocked/in-process — no services need to be running):
```sh
python -m pytest -v
```

**End-to-end smoke test** (starts all 4 real services against real data, exercises every
critical path over HTTP — fraud SHAP, ledger timeline, typology network, full agent
investigation for all 3 tiers, model-backed counterfactual, chat with cached fallback — then
tears everything down):
```sh
python scripts/smoke_test.py
```

**Unified Agent Rigor Evaluation Harness (Grounding + Narrative-Model Consistency):**
```sh
python scripts/evaluate_agent.py
```

## 📏 Fraud Score Uncertainty Quantification

The fraud engine's LightGBM risk score is wrapped in a split-conformal prediction interval
(`engines/fraud/conformal.py`, built with [`mapie`](https://mapie.readthedocs.io)), additive to
the existing SHAP factors panel. For a flagged transaction the dashboard shows, e.g.:

> Risk score **1.00**, with a **90%** confidence interval of **[0.94, 1.00]** (split-conformal,
> validated at **90.3%** empirical coverage)

**Measured empirical coverage** (does the true label fall inside the interval ~90% of the time,
on a held-out evaluation set disjoint from calibration): **90.28%** overall, against a 90%
target, on 28,481 evaluation rows from `data/raw/creditcard.csv`. Calibration partitions by the
model's own predicted verdict (Mondrian conformal) rather than a single pooled quantile — a
pooled quantile on this ~99.83%-legitimate dataset gave a technically-valid ~90% marginal
coverage number from an interval so narrow it achieved 0% coverage specifically on fraud rows;
partitioning fixed that (fraud-class coverage 0% → 72.7%). Full derivation, the failure mode
found, and per-partition numbers: [`ARCHITECTURE.md` §6](ARCHITECTURE.md).

Runs automatically on first bring-up (`scripts/dev_up.py`); re-run manually after retraining
the model:
```sh
python -m engines.fraud.conformal
```

---

## 🧪 Agent Rigor & Evaluation Metrics

Verity evaluates agent reasoning fidelity using two complementary, code-level metrics:

| Metric | Result | Operational Scope & Caveat |
| :--- | :--- | :--- |
| **Grounding Coverage** (Anti-Hallucination) | **100%** unverified claims blocked (66.7% retention) | Code-level sentence-by-sentence fact-extractor purging hallucinated claims unbacked by tool outputs. |
| **Narrative-Model Consistency** | **83.3%** *(10/12 passed)* | **REAL-LLM ONLY CAVEAT:** Evaluated under live LLM reasoning (`VERITY_LLM_API_KEY` set). If run in deterministic/rule-based mode, this reports `N/A — deterministic mode, result is not meaningful` because deterministic fallback constructs sentences directly from SHAP factors, yielding trivial 100% agreement by construction. |

### What Narrative-Model Consistency Measures
Narrative-model consistency measures the percentage of counterfactual transaction perturbations where an independently generated LLM narrative's verdict and named primary drivers strictly agree with the re-run LightGBM risk score (relative to calibrated threshold $\tau = 0.8843$) and top 1–2 SHAP attributions.

### How it Complements Grounding Coverage
- **Grounding Coverage** checks whether the agent claims something it *cannot prove* (e.g., hallucinated offshore bank accounts or fabricated merchant IDs).
- **Narrative-Model Consistency** checks whether what the agent claims *remains mathematically true* when the underlying facts and features change.

### Scope & Future Roadmap
*Hackathon Scope Constraint:* This benchmark is intentionally scoped to a fixed suite of 12 targeted perturbations across the 3 demo fraud cases reusing the existing counterfactual re-run pipeline (`engines/fraud/explain.py`). A full production implementation would incorporate automated minimum-plausible-distance perturbation search (e.g., Nelder-Mead / Tree-SHAP boundary walk) and semantic embedding drift analysis across the entire transaction distribution.

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


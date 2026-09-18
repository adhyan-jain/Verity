# Getting Started — Verity from a Fresh Clone

Step-by-step for someone who has never touched this repo before. Every step here was run
against a real fresh `git clone` of this branch during the integration pass — not just read
off the code. For the condensed version see the root [`README.md`](README.md); for what's
actually wired together and why see [`ARCHITECTURE.md`](ARCHITECTURE.md).

## 0. Prerequisites

| Tool | Version | Check |
|---|---|---|
| Python | 3.11+ | `python --version` |
| Node.js | 18+ | `node --version` |
| npm | (ships with Node) | `npm --version` |
| Git LFS | any recent | `git lfs version` |

`data/raw/creditcard.csv` (~144MB) and `data/raw/bank.xlsx` (~6MB) are stored via **Git LFS**.
If `git lfs` isn't installed when you clone, those two files will be tiny text pointers instead
of real data and every engine will fail to start. Install Git LFS first
(https://git-lfs.com), then `git lfs install` once per machine, *before* cloning — or run
`git lfs pull` after the fact if you already cloned without it.

## 1. Clone

```sh
git clone https://github.com/adhyan-jain/Verity.git
cd Verity
git lfs pull   # only needed if git lfs wasn't installed at clone time
```

Verify the datasets are real files, not LFS pointers:
```sh
ls -la data/raw/creditcard.csv data/raw/bank.xlsx
# creditcard.csv should be ~144MB, bank.xlsx ~6MB — a few hundred bytes means LFS didn't pull.
```

## 2. Install dependencies

**Python** (all four backend services share one virtualenv):
```sh
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

**Dashboard** (npm only — see ARCHITECTURE.md §4.1.7 for why not bun):
```sh
cd dashboard
npm install
cd ..
```

## 3. Configure environment

```sh
cp .env.example .env
```

The defaults in `.env.example` work as-is for local development (all four services on
`localhost:8000-8003`, dashboard on `localhost:3000`, a placeholder `FRAUD_API_KEY`). You only
need to edit it if you're changing ports, pointing at a non-local engine, or wiring in a real
external LLM (`VERITY_LLM_API_KEY`) — none of which are required to run the full stack.

## 4. Run everything

One command starts all four backend services and the dashboard:
```sh
python scripts/dev_up.py
```

**First run only**: `engines/fraud/model.pkl` doesn't exist yet (it's gitignored — a trained
binary artifact, not source), so the script trains it automatically before starting the fraud
service. This takes **~90 seconds** on a normal machine and only happens once; subsequent runs
skip straight to starting services. You'll see:
```
[setup] engines/fraud/model.pkl missing — training fraud model (~2 min)...
...
[start] agent -> http://127.0.0.1:8000
[start] fraud -> http://127.0.0.1:8001
[start] ledger -> http://127.0.0.1:8002
[start] typology -> http://127.0.0.1:8003
[ready] agent healthy
[ready] fraud healthy
[ready] ledger healthy
[ready] typology healthy
[start] dashboard -> http://localhost:3000
```

Open **http://localhost:3000**. Ctrl+C stops everything (backend services + dashboard).

Backend-only (skip the dashboard dev server, e.g. if you're developing the frontend separately
or driving the APIs directly):
```sh
python scripts/dev_up.py --no-dashboard
```

### First real request per engine is slower than the rest
`engines/ledger/api.py` parses the full `bank.xlsx` (116k rows) on its first
`/accounts`/`/timeline`/`/anomalies` call and caches the result to `data/cache/` — expect the
very first ledger request after startup to take up to ~30-45s, with every request after that
near-instant. This is a one-time-per-process cost, not a bug (see ARCHITECTURE.md §5.3).

## 5. Verify it actually works

**Unit/integration tests** (fast, no services need to be running — mocks the network):
```sh
pytest -v
```
Expect `56 passed`.

**End-to-end smoke test** (starts all 4 real services against real data, exercises every
critical path over real HTTP — fraud auth + SHAP, ledger accounts/timeline/walk, typology
network/flags, full agent investigation for all 3 tiers, model-backed counterfactual, chat with
disclosed cached fallback — then shuts everything down):
```sh
python scripts/smoke_test.py
```
Expect `SMOKE TEST PASSED` and exit code 0. This takes ~30-40s if `model.pkl` already exists,
~2 minutes on the very first run (trains the model first).

**Dashboard build:**
```sh
cd dashboard
npx tsc --noEmit    # typecheck
npm run build        # full production build (client + SSR + nitro)
```

## 6. Manual / per-service runs (debugging one piece in isolation)

```sh
python -m uvicorn agent.api:app --port 8000
python -m uvicorn engines.fraud.api:app --port 8001
python -m uvicorn engines.ledger.api:app --port 8002
python -m uvicorn engines.typology.api:app --port 8003
npm --prefix dashboard run dev
```
All of these read from the same `.env` at the repo root — but only when started via
`scripts/dev_up.py`, which has its own small `KEY=VALUE` parser
(`scripts/_services.py::load_env_file`, no external dependency). A bare `uvicorn` command does
**not** load `.env` on its own; export the vars in your shell yourself (or use `direnv`/
`dotenv-cli`) if you're running a single service manually like this.

## 7. Common problems

| Symptom | Cause | Fix |
|---|---|---|
| Fraud engine `503 Service misconfigured` | `FRAUD_API_KEY` not set in the fraud process's environment | Make sure `.env` exists and `dev_up.py`/your shell actually exports it |
| Fraud engine `401` from the agent/dashboard | `FRAUD_API_KEY` and `VITE_FRAUD_API_KEY`/agent's `FRAUD_API_KEY` don't match | Use the same value everywhere in `.env` |
| `FileNotFoundError: ... model.pkl` | Trying to hit the fraud engine directly without training first | Run `python scripts/dev_up.py` once (auto-trains), or `python -m engines.fraud.train` directly |
| Ledger `/timeline` or `/accounts` hangs for 10-45s then works | First-call Excel parse (see step 4) | Not a bug — wait it out once per process restart |
| `Address already in use` on port 8000-8003/3000 | A previous run's process didn't exit cleanly | Kill the stale process for that port, or restart your terminal |
| Dashboard shows "Offline fixtures active" | Backend services aren't reachable from the browser | Confirm all 4 services are healthy (`curl http://localhost:8000/health` etc.) and `VITE_*_API_URL` in `.env` match where they're actually running |

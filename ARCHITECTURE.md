# Verity — Integration Architecture & Gap Report

Written during the integration pass (`integration/wiring` branch). This documents the
system as it actually exists after three parallel builders (fraud engine / ledger+typology
engine / agent core) plus a fourth, separately-generated Lovable UI export, and lists every
gap found before any wiring changes are made. Big changes wait for sign-off on this doc.

## 1. Components as built

| Component | Owner (per docs) | Language / framework | Port | Entry point | Status found |
|---|---|---|---|---|---|
| `engines/fraud` | Person A | Python 3.11, FastAPI, LightGBM, SHAP | 8001 | `engines/fraud/api.py` (`uvicorn engines.fraud.api:app`) | Code complete. `model.pkl` is gitignored and **not present** — `train.py` was never run in this checkout. 5/6 failing tests trace to this. |
| `engines/ledger` | Person B | Python 3.11, FastAPI, pandas | 8002 | `engines/ledger/api.py` | Complete, tests pass. Reads `data/raw/bank.xlsx` (present via Git LFS). |
| `engines/typology` | Person B | Python 3.11, FastAPI, NetworkX | 8003 | `engines/typology/api.py` | Complete, tests pass. Generates `data/synthetic/synthetic_network.json` on first run if absent. |
| `agent` | Person C | Python 3.11, FastAPI (hand-rolled loop, no framework) | 8000 | `agent/api.py` | Complete, 49/50 tests pass. One real bug (§3.6). |
| `dashboard` | Person D (Lovable export) | TypeScript, React 19, TanStack Start/Router, Vite | 3000 (Vite/TanStack Start default) | `dashboard/src/routes/index.tsx` → `VerityWorkspace` | **Does not build.** Imports an entire `src/lib/` directory that was never committed (§3.1). Also hardcodes all case/evidence data — nothing is actually wired to any backend. |
| `contracts/schemas.json` | All (agreed hour 1) | JSON Schema | — | — | Present, used as the intended single source of truth. Engines mostly, but not perfectly, conform (§3.3). |

### Intended end-to-end flow
1. Analyst (Priya) opens the dashboard, sees a **unified case queue** mixing card-fraud and
   ledger/typology alerts.
2. Selecting a case shows the primary transaction, SHAP factors (fraud) or timeline/anomalies
   (ledger) or graph hops (typology).
3. Analyst triggers an **agent investigation** (`POST /api/v1/agent/investigate`): the agent
   loop calls the 4 quarantined tools (`get_transaction`, `get_shap_explanation`, `walk_graph`,
   `counterfactual`), which in turn call the three engine APIs over HTTP (`VERITY_ENV=live`) or
   fall back to `contracts/mock_data/` fixtures (`VERITY_ENV=mock`, the default).
4. Every tool call produces an `AgentTraceEvent`; `agent/grounding.py` strips any narrative
   sentence not backed by a real trace event before it reaches the UI.
5. Analyst can ask free-form questions or counterfactuals through the chat/query panel, which
   hits `POST /api/v1/agent/chat` or `/counterfactual`; slow (>20s) or unmatched queries fall
   back to pre-baked answers in `contracts/mock_data/mock_fallback_qa.json`, with an explicit
   "using a prepared answer" notice (never silent).

That flow is coherent on paper. In practice, **nothing below the dashboard is called**, and the
dashboard itself doesn't compile, so today there is no working end-to-end path at all.

## 2. Who-built-what fingerprints

- **Person A (fraud)**: strict, defensive style — API-key auth (`X-API-Key`/`FRAUD_API_KEY`,
  fail-closed), checksum-verified model artifacts, explicit `require_ready` guards. Heaviest
  dependency footprint (`lightgbm`, `shap`, `imbalanced-learn`).
- **Person B (ledger/typology)**: no auth, `allow_origins=["*"]` CORS, in-memory caches,
  `try/except (ImportError, ValueError)` dual-import shims for running standalone vs. as a
  package — same shim pattern repeated verbatim in both `engines/ledger/api.py` and
  `engines/typology/api.py`, evidence of one author.
- **Person C (agent)**: also `allow_origins=["*"]`, dual-mode `VERITY_ENV=mock|live` with silent
  fallback-to-mock on any request exception, hand-rolled everything (no LangChain et al., per
  their own playbook's "Law 1").
- **Person D (dashboard)**: a Lovable.dev-generated TanStack Start app, own `README.md` still
  says "Radiant Canvas" / points at a Lovable-hosted URL, own `.gitignore`, own `bunfig.toml`
  *and* `package-lock.json` (two lockfiles — someone ran both `bun install` and `npm install`).
  Never integrated with the other three services at all; `verity-workspace.tsx` has all case
  data hardcoded in a local `CASES` array.

## 3. Integration problems found

### 3.1 Dashboard doesn't build — `src/lib/` was never committed
Root cause: the **root `.gitignore`** (Python-oriented, written by Person A/B) has a bare
`lib/` entry meant to ignore Python's `venv/lib/`. Git applies that pattern repo-wide, so it
also silently swallowed `dashboard/src/lib/**` — a directory shadcn/TanStack Start scaffolding
requires. `git log --all` confirms these files were never added in any commit:
- `dashboard/src/lib/utils.ts` — the `cn()` helper, imported by all 47 files under `components/ui/`.
- `dashboard/src/lib/api-client.ts` — imported by `verity-workspace.tsx`
  (`checkEnginesHealth`, `askCounterfactualOrChat`, `fetchLiveTimeline`,
  `fetchLiveTypologyNetwork`, `EngineStatus`). **Never existed at all**, not just gitignored —
  the dashboard was built entirely against a hardcoded fixture, and this file is a stub that
  was designed against but never written.
- `dashboard/src/lib/error-capture.ts`, `error-page.ts` — imported by `server.ts`/`start.ts` SSR
  error boundary.
- `dashboard/src/lib/lovable-error-reporting.ts` — imported by `routes/__root.tsx`.

Fix: scope the ignore rule (`engines/**/lib/`, `**/venv/lib/`, or similar) so it can't shadow
`dashboard/src/lib/`, and write the missing files. `api-client.ts` needs to be written fresh
against the real engine/agent contracts (§4).

### 3.2 Dashboard has zero real wiring regardless of the missing files
Even once `src/lib` exists, `verity-workspace.tsx`'s `CASES`, SHAP factors, timeline points,
and network diagram are all inline fixtures — not fetched from anywhere. `checkEnginesHealth`
is called (for the masthead's online/offline indicator) but nothing else in the file consumes
live data. This needs real fetch wiring, not just a missing-file fix.

### 3.3 API contract mismatches between spec/docs and actual engine code
- `docs/API_SPEC.md` documents `GET /api/v1/ledger/anomalies/{account_id}` and
  `GET /api/v1/ledger/timeline/{account_id}` — these exist. But it omits
  `GET /api/v1/ledger/accounts`, `GET /api/v1/ledger/walk/{account_id}`, and
  `GET /api/v1/ledger/transaction/{transaction_id}`, all of which exist in `engines/ledger/api.py`
  and are the ones `agent/tools.py` actually needs (`/walk/{account_id}`). Docs are stale, not
  code — no fix needed to code, but `docs/API_SPEC.md` should be refreshed in cleanup.
- `docs/API_SPEC.md` never mentions the typology engine's `GET /api/v1/typology/network` or
  the fraud engine's `X-API-Key` requirement at all — an analyst/integrator following the doc
  would hit silent 401/503s.
- `agent/tools.py` (`VERITY_ENV=live`) calls `FRAUD_API_URL/transaction/...` and
  `FRAUD_API_URL/explain/...` **without ever sending `X-API-Key`**. Since
  `engines/fraud/api.py` fails closed (`require_api_key`), every live call from the agent to
  the fraud engine will 401 (or 503 if the env var is simply unset) and silently fall back to
  mock data — masking a real integration failure as if it were "engine offline." This is the
  most severe live-mode bug: two Person-A/Person-C-owned modules never agreed on auth.
- Everything else (`ReconciliationAnomaly`, `GraphWalkStep`, `TypologyFlag`, `AgentTraceEvent`)
  matches `contracts/schemas.json` field names/casing (snake_case throughout — good, no
  camelCase/snake_case split exists in the backend). The frontend never got far enough to
  diverge.

### 3.4 CORS
- `engines/ledger/api.py`, `engines/typology/api.py`, `agent/api.py`: `allow_origins=["*"]`
  (fine for dev, permissive for anything later).
- `engines/fraud/api.py`: `allow_origins=_ALLOWED_ORIGINS`, sourced from `FRAUD_CORS_ORIGINS`
  env var, **empty by default** → blocks every browser origin, including the dashboard's own
  `/health` check, until the env var is set. Nothing currently sets it.

### 3.5 Duplicate/competing package managers in `dashboard/`
`bun.lock` **and** `package-lock.json` **and** `bunfig.toml` are all present and committed.
Two different install tools were used at different points; only one should be canonical or a
clean install can silently pick up drifted resolutions between the two lockfiles.

### 3.6 Real bug: latency-guard fallback flaky on Windows
`agent/fallback.py::execute_with_latency_guard` times the task with `time.time()` (coarse
resolution on Windows, can return **identical** values across a fast synchronous call) and
compares with strict `>`. `tests/test_agent_fallback.py::test_latency_guard_timeout` (forcing
`timeout_seconds=0.0`) fails nondeterministically on this platform because `elapsed > 0.0` can
be `False` when `elapsed` measures as exactly `0.0`. Needs `time.perf_counter()` + `>=`.

### 3.7 Missing model artifact
`engines/fraud/model.pkl` is gitignored (correctly — it's a binary artifact) but was never
generated in this checkout, and no setup script runs `train.py`. `engines/fraud/explain.py`
and 5 of `tests/test_fraud_engine.py`'s tests hard-fail without it. This is expected
first-run state, not a design flaw, but needs to be part of the one-command bring-up.

### 3.8 No unified run command, no root `.env`/`.env.example`
Each engine reads its own env vars (`FRAUD_API_KEY`, `FRAUD_CORS_ORIGINS`, `VERITY_ENV`,
`FRAUD_API_URL`, `LEDGER_API_URL`, `TYPOLOGY_API_URL`, `AGENT_TOOL_TIMEOUT`,
`VERITY_LLM_API_KEY`/`OPENAI_API_KEY`, `VERITY_LLM_BASE_URL`, `VERITY_LLM_MODEL`) with no
central place they're documented together, and no `.env.example` despite the root
`.gitignore` already special-casing one (`!.env.example`). Four independent `uvicorn`
processes plus a Vite dev server currently have no single start command.

### 3.9 Dead code / stubs
- `agent/llm.py` always falls back to `_builtin_reasoning` unless a real
  `VERITY_LLM_API_KEY`/`OPENAI_API_KEY` is configured — by design (documented dual-mode), not a
  bug, but worth calling out: "LLM-backed" reasoning is a rules-based stand-in until a key is
  supplied.
- `dashboard/src/components/verity-workspace.tsx`'s `EvidenceNetwork`/`EvidenceTimeline` render
  fixed SVG paths / synthetic points regardless of the selected case — cosmetic placeholders,
  not wired to `fetchLiveTimeline`/`fetchLiveTypologyNetwork` even after those functions exist.
- `docs/PITCH.md`, `DEMO_SCRIPT.md` reference a live demo flow that assumes all of the above
  already works.

## 4. Plan to close the gaps (proposed, not yet applied)

1. **Fix the gitignore collision**; add `dashboard/src/lib/*.ts` (utils, error-capture,
   error-page, lovable-error-reporting, api-client) for real, written against the actual
   engine/agent contracts above.
2. **Fix the fraud API-key gap**: plumb `FRAUD_API_KEY` through `agent/tools.py`'s live-mode
   requests (`X-API-Key` header) and through the dashboard's direct health check.
3. **Root `.env.example`** covering every env var above, with sane localhost defaults
   (ports 8000-8003, `FRAUD_CORS_ORIGINS=http://localhost:3000`, `VERITY_ENV=live` for the
   integrated run).
4. **One bring-up path**: a small script/`docker compose`/`package.json`-level orchestration
   that (a) trains the fraud model if `model.pkl` is missing, (b) starts all 4 uvicorn services,
   (c) starts the dashboard dev server — plus a smoke-test script exercising
   health → investigate → chat → counterfactual end to end.
5. **Pick one dashboard package manager** (npm, since `package-lock.json` + `package.json`
   scripts are the canonical TanStack Start setup and npm is already required by the root
   `README.md` prerequisites) and remove the other lockfile.
6. **Fix the Windows timing bug** in `agent/fallback.py`.
7. Refresh `docs/API_SPEC.md` to match actual routes (cleanup phase, after everything runs).

None of this touches the actual detection logic (SHAP, anomaly rules, FATF typology rules,
grounding filter) — those are complete and tested. The gap is entirely in wiring, auth, config,
and one missing frontend directory.

**Open question for you before I proceed**: dashboard currently only wires a masthead health
check to live data; case list/evidence/factors are fixtures. Do you want me to (a) keep the
fixture `CASES` array as *seed* data but make evidence panels (SHAP factors, timeline, graph)
fetch live from the engines for the currently-selected case, replacing the placeholder SVG/points,
or (b) leave the dashboard's visual data as-is and only wire the parts already stubbed
(`checkEnginesHealth`, `askCounterfactualOrChat`) since that's the minimum to make the imports
resolve and the query panel functional? I'm defaulting to (a) — it's what "replace mocks with
real calls" in the brief calls for — unless you'd rather scope it down.

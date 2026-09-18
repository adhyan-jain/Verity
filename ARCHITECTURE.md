# Verity — Integration Architecture & Gap Report

Written during the integration pass (`integration/wiring` branch). Sections 1–3 are the
as-found survey from before any changes (kept for the record — component map, who-built-what
fingerprints, and every gap found). **Section 4 records what was actually fixed and verified**;
section 5 lists what's still open for a human decision. Everything below was run for real —
trained the model, started all 4 services, drove the dashboard in a real browser, see §4.5.

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

## 4. What was fixed, and how it was verified

### 4.1 Fixes applied (each is its own commit on `integration/wiring` — see git log)
1. **Gitignore collision** — scoped the root `.gitignore`'s bare `lib/`/`lib64/` rule so it
   can no longer shadow `dashboard/src/lib/**`; Python venv dirs are already covered by the
   `venv/`/`.venv/` patterns below it, so nothing was lost.
2. **Wrote the missing `dashboard/src/lib/*.ts`** for real: `utils.ts` (`cn()`), `api-client.ts`
   (against the actual engine/agent contracts, not the never-realized stub), `error-capture.ts`,
   `error-page.ts`, `lovable-error-reporting.ts`. Dashboard now typechecks (`tsc --noEmit`),
   lints clean, and builds (`vite build`, SSR + client + nitro all succeed).
3. **Fraud API-key gap** — `agent/tools.py` now forwards `X-API-Key: $FRAUD_API_KEY` on every
   live-mode call into the fraud engine (`get_transaction`, `get_shap_explanation`,
   `counterfactual`). Verified: `scripts/smoke_test.py` step 1b asserts an unauthenticated
   direct call is rejected (401) while the agent's own live calls succeed.
4. **`get_transaction` tier-routing bug (found during live wiring, not in the original
   survey)** — in live mode, `get_transaction` unconditionally called the fraud engine
   regardless of the transaction's actual tier. Since the fraud engine parses any digits out
   of the ID as a row index, a ledger or synthetic-network ID (e.g. `TX-SYNTH-5501`) would
   silently resolve as some unrelated `real_card` row, and every downstream reasoning step
   (`get_shap_explanation` instead of `walk_graph`) followed the wrong tier. Fixed by routing
   on ID prefix (`LEDGER` → ledger engine, `SYNTH`/`SYN` → typology network edge lookup, else
   → fraud), matching the same prefix logic already used by the mock-mode fallback in the same
   function. Regression test: `tests/test_agent_tools.py::test_live_mode_routes_get_transaction_by_id_prefix`;
   also covered live end-to-end by `scripts/smoke_test.py` step 4.
5. **Windows latency-guard timing bug** — `agent/fallback.py::execute_with_latency_guard` now
   uses `time.perf_counter()` (monotonic, high-resolution) and `>=` instead of `time.time()` +
   strict `>`, which could tie at exactly `0.0` elapsed on Windows and skip the fallback path
   entirely. `tests/test_agent_fallback.py::test_latency_guard_timeout` now passes reliably.
6. **Root `.env.example`** — every env var from every service in one file, with working
   localhost defaults, including `FRAUD_CORS_ORIGINS=http://localhost:3000` (previously unset,
   which blocks every browser origin by design — the fraud engine fails closed on CORS too).
7. **Dropped the duplicate package manager** — removed `dashboard/bun.lock` and
   `dashboard/bunfig.toml`; `package.json` + `package-lock.json` (npm) is now the only
   canonical install path, matching the root README's stated prerequisite.
8. **Wired the dashboard to live data** (`dashboard/src/components/verity-workspace.tsx`):
   - Reasoning trace + risk score: calls `POST /api/v1/agent/investigate` per selected case;
     renders real `AgentTraceEvent`s + narrative when it succeeds, falls back to the labeled
     fixture trace on any failure (never a blank panel).
   - Risk factors: live SHAP `top_factors` from the fraud engine for the `real_card` case;
     labeled fixture for the other two tiers (see §5.1 for why those can't resolve live yet).
   - Evidence surface: live ledger timeline (first real account from `/ledger/accounts`) and
     live typology network (`/typology/network`, real node/edge counts and amounts) replace the
     decorative SVG's text/labels when available; the geometry stays the same, only the data
     behind it is real.
   - Every live surface is explicitly labeled "Live ..." vs "Demo exhibit" — the product's own
     honesty principle (`VERITY_BUILD_SPEC.md` §6) applied to the wiring itself, not just the
     chat fallback it was written for.
9. **One-command bring-up + real smoke test** — `scripts/dev_up.py` (trains the model if
   missing, starts all 4 services + dashboard) and `scripts/smoke_test.py` (starts all 4 real
   services, no mocks, exercises every critical path over HTTP, tears down, exit code reflects
   pass/fail).

### 4.2 What was deliberately left alone
Per the brief's "don't rewrite working logic just for style" — SHAP computation, anomaly
detection math, FATF typology rules, the grounding filter, and the hand-rolled agent loop are
untouched. All fixes above are wiring, auth, config, or a genuine cross-tier routing bug; none
touch detection logic.

### 4.3 Naming/casing/error-format conventions
No unification was needed here: every Python service already used snake_case field names
matching `contracts/schemas.json` verbatim, and FastAPI's default `{"detail": "..."}` error
shape is consistent across all four services. The frontend never got far enough to diverge
before this pass. The one real convention decision was the package manager (§4.1.7) — picked
npm over bun because it required deleting one lockfile instead of migrating `package.json`
scripts, tooling configs, and CI expectations to bun.

### 4.4 Clean install, from scratch
Verified in this session: `python -m venv .venv && pip install -r requirements.txt` and
`cd dashboard && npm install` both succeed from a clean checkout with no manual patching.

### 4.5 End-to-end verification actually performed
- `pytest -v` → **56 passed**, 0 failed (was 6 failed / 49 passed at survey time).
- `python scripts/smoke_test.py` → **all checks passed**: fraud auth + SHAP, ledger accounts/
  timeline/walk against the real 10-account `bank.xlsx`, typology network/flags against the
  real generated FATF graph, full agent investigation loop for all three tiers (verified each
  stays on its correct `tier_origin` and calls the correct tool), model-backed counterfactual
  (verified the recalculated score actually moves on a real fraud-labeled row), chat with
  disclosed cached fallback.
- `cd dashboard && npx tsc --noEmit && npm run build` → clean typecheck, successful client + SSR
  + nitro build.
- Drove the running dashboard in a real headless browser against all 4 live services: masthead
  reports "Engines online (4/4)"; selecting the synthetic-network case shows a live-fetched
  network caption ("Live network · 29 accounts · 192 transactions") and a reasoning trace with
  real `get_transaction` → `walk_graph` events (this is what caught the tier-routing bug in
  §4.1.4 — it only surfaced once the dashboard was actually driving the real loop).

## 5. Remaining gaps — needs a human decision or follow-up work

### 5.1 Demo case IDs don't resolve against the live datasets (needs a decision)
`contracts/mock_data/` and the dashboard's fixture `CASES` array use illustrative IDs
(`ACC-1092`, `TX-LEDGER-3011`, `ACC-SYN-401`, `TX-SYNTH-5501`) invented for the mock-mode demo
narrative. The real datasets don't share that ID space:
- Real ledger account IDs are literal bank account numbers (e.g. `409000362497`), and real
  parsed transaction IDs are sequential (`TX-LEDGER-000001` post-sort) — nothing like `ACC-1092`.
- The real generated synthetic network seeds role-based IDs (`ACC-SMURF-*`, `ACC-RT-*`,
  `ACC-LAYER-*`, `ACC-BENIGN-*`) and edge IDs like `TX-SYNTH-0020` — nothing like `ACC-SYN-401`
  or `TX-SYNTH-5501`.
- Net effect: in live mode, the ledger/synthetic demo cases' `get_transaction`/`walk_graph`
  calls correctly fail to match anything live (by design — this is honest behavior, not a
  crash: `walk_graph` on a nonexistent live account returns "no connected entities" rather than
  fabricating a path) and fall back to the mock fixture, same as before this pass. Only the
  `real_card` case (`TX-CARD-9842`, which the fraud engine parses as a literal, in-range row
  index) resolves against real data for every tool.
- **Decision needed**: either (a) regenerate `contracts/mock_data/` and the dashboard's `CASES`
  fixture from IDs actually present in the live datasets (requires picking specific real
  accounts/rows to build a demo narrative around, and touches Person B's and Person D's
  content), or (b) accept that only the card-fraud case is fully live-representative and the
  other two intentionally demonstrate the mock fallback path. Not changed in this pass — it's
  a product/demo-content decision, not a code defect.

### 5.2 `VITE_FRAUD_API_KEY` ships inside the client bundle
Documented in `.env.example` and `api-client.ts`: embedding the fraud engine's API key in a
Vite client build means it's visible to anyone who opens the bundle. Acceptable for this
decision-support prototype behind an internal network (per `VERITY_BUILD_SPEC.md` §8, "not a
production-audited compliance system"); a real deployment should proxy fraud-engine calls
through the TanStack Start server (`dashboard/src/server.ts` already exists as that seam) so
the key never reaches the browser.

### 5.3 First ledger `/timeline/{account_id}` call is slow for large accounts
No pre-warmed cache (`data/cache/timelines.json` is only ever read, never written by anything
in this repo) — the first request for a given account computes anomalies on demand. Measured
up to ~15-20s for the largest account in this dataset. Not a correctness issue (the smoke test
timeout was raised to accommodate it), but worth a pre-warm step or a longer client timeout if
this becomes the default landing view.

### 5.4 `docs/API_SPEC.md` was stale; now refreshed
Was missing `/ledger/accounts`, `/ledger/walk`, `/ledger/transaction`, `/typology/network`, the
fraud engine's auth requirement, and the agent's own API entirely. Rewritten in this pass to
match the implementation, verified against `scripts/smoke_test.py`'s real calls.

### 5.5 Untouched, still true from the original survey
- `agent/llm.py`'s "LLM-backed" reasoning is a deterministic rules engine unless
  `VERITY_LLM_API_KEY`/`OPENAI_API_KEY` is supplied — by design, not a gap, but worth restating:
  nothing in this pass required or tested a real external LLM call.
- The duplicate `POST /api/v1/agent/counterfactual` route registered directly on the fraud
  engine (`engines/fraud/api.py`) is dead code — nothing calls it; the agent's own
  `/api/v1/agent/counterfactual` (which goes through `agent/tools.py`) is what's actually used
  end-to-end. Flagging rather than deleting, since it's Person A's code and not blocking
  anything — a call to remove it belongs to a future cleanup pass with their sign-off.

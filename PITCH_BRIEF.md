# Verity — Pitch Briefing

Written for whoever is presenting, not for engineers. Every claim below is grounded in the
actual codebase and was checked by running it — not by reading a teammate's comment and trusting
it. Where something is unfinished or mocked, this doc says so directly, because that's more
useful to you on stage than getting caught by a judge's question.
Use this brief as the source of truth for the live demo narrative.

---

## 1. One-liner

**Verity is one workspace where a bank analyst can see every fraud and money-laundering alert
in one queue, and ask an AI agent to explain — with receipts — exactly why each one was
flagged.**

---

## 2. The problem

Mid-size digital banks and NBFCs (non-bank lenders) are too small to run separate fraud and
anti-money-laundering (AML) teams. One analyst — the persona here is "Chitrita" — ends up
swivel-chairing between a card-fraud dashboard and a separate AML/ledger review tool, with two
different alert queues, two different mental models, and no single place to ask "why did the
system flag this?" and get a trustworthy answer.

The deeper problem the AI framing solves: analysts don't trust black-box ML flags, and they
*really* don't trust chatbot summaries that might be quietly making things up (a hallucinated
"offshore account" claim in a compliance report is a real liability, not a UX nitpick). Verity's
answer is an AI agent that is architecturally forbidden from saying anything it can't point to
a specific data lookup for.

**Who has this problem:** compliance/fraud analysts at smaller financial institutions — not
Tier-1 banks with dedicated teams for each alert type, but the much larger number of
mid-size/digital-first lenders who can't staff that way.

---

## 3. How it works — user view (walk it like a demo)

1. **Chitrita opens the workspace.** One screen. A masthead shows whether the four backend
   services are live ("Engines online (4/4)") — this is a real health check, not decoration.
2. **She sees a case queue** mixing three kinds of alerts side by side: a card-fraud
   transaction, a bank-ledger anomaly, and a synthetic money-laundering network — filterable by
   type, searchable by ID.
3. **She clicks a case.** The screen updates with:
   - A **risk score** (0.00–1.00) and a one-line summary.
   - An **evidence surface** — either a horizontal timeline of that account's real transactions
     with anomaly windows highlighted, or a network graph of the accounts involved.
   - A **risk-factors panel** — for the card case, the actual SHAP model attribution (which
     features drove the score, split into "we can explain this" vs. "this is a mathematical
     signal we can't turn into a human story").
4. **Behind the scenes, an AI agent has already investigated the case.** It called real data
   tools (never guessed), and a **reasoning trace** panel shows each step it took, in order,
   each one timestamped and tagged with which tool it called — culminating in a narrative
   paragraph built *only* from those verified steps.
5. **She asks a follow-up question** in the "interrogate the evidence" panel — e.g. *"what if
   the amount were $150 instead?"* The system doesn't guess; it **re-runs the actual fraud
   model** with the changed number and shows the new score. Ask something unrelated and it
   either answers from real case context or, if that's slow (>20 seconds) or unmatched, falls
   back to a pre-verified answer — and says so explicitly on screen. It never silently makes
   something up.
6. **She marks the case** "Escalate" or "Needs review" and moves to the next one.

---

## 4. How it works — technical view

Four independent backend services plus one frontend, talking over plain HTTP/JSON:

```
                         ┌─────────────────────────┐
                         │   Dashboard (Browser)    │
                         │  React + TanStack Start  │
                         │      localhost:3000      │
                         └───────────┬──────────────┘
                                     │ HTTP (fetch)
              ┌──────────────────────┼───────────────────────┐
              │                      │                        │
              ▼                      ▼                        ▼
    ┌──────────────────┐  ┌───────────────────┐   ┌─────────────────────┐
    │  Fraud Engine     │  │  Agent Core        │   │  Ledger Engine      │
    │  FastAPI :8001    │  │  FastAPI :8000     │   │  FastAPI :8002      │
    │  LightGBM + SHAP  │◄─┤  hand-rolled loop  │   │  pandas             │
    │  (API-key auth)   │  │  (no LangChain)    │   └──────────┬──────────┘
    └───────────────────┘  │                    │              │
                            │                    │◄─────────────┘
                            │                    │
                            │                    │   ┌──────────────────────┐
                            │                    ├──►│  Typology Engine      │
                            │                    │   │  FastAPI :8003        │
                            │                    │   │  NetworkX (graph)     │
                            └────────────────────┘   └───────────────────────┘
```

**The five pieces, plainly:**

| Piece | What it does | Tech |
|---|---|---|
| **Fraud Engine** | Scores individual card transactions for fraud risk, explains *why* via SHAP | Python, FastAPI, LightGBM (a gradient-boosted tree model) |
| **Ledger Engine** | Parses a real bank's transaction history, flags balance drops / velocity spikes / suspicious reversals per account | Python, FastAPI, pandas |
| **Typology Engine** | Detects money-laundering *patterns* (not single transactions) across a network of accounts — structuring, round-tripping, rapid layering — using formal FATF definitions | Python, FastAPI, NetworkX (graph algorithms) |
| **Agent Core** | The "brain." Runs a step-by-step investigation loop, calls the three engines above as tools, and — critically — strips out any sentence in its final narrative that isn't backed by a real tool call | Python, FastAPI, no AI framework (deliberately hand-written) |
| **Dashboard** | The UI Chitrita actually uses | TypeScript, React 19, TanStack Start (server-rendered React), Vite |

**How they talk:** everything is plain REST/JSON over `localhost` ports 8000–8003 in
development. The dashboard calls the Agent for investigations/chat, and calls the Ledger/
Typology engines directly for the visual evidence panels (timeline, network graph) — the fraud
engine is API-key-protected, so the dashboard only calls it directly for its public health
check.

---

## 5. The clever parts

- **The agent is not allowed to lie, enforced in code, not in a prompt.** Most "AI agent"
  demos say "we told the model not to hallucinate" in the prompt and hope. Verity's
  `agent/grounding.py` is a second, independent piece of code that takes the AI's draft answer,
  splits it into sentences, and **deletes any sentence** that contains a claim not traceable
  to an actual tool-call result — including a specific test for exactly the kind of thing that
  makes compliance officers nervous (an AI inventing an "offshore account" or "shell company"
  that was never actually found).
- **No agent framework.** The investigation loop (`agent/loop.py`) is a plain, ~250-line
  Python loop: ask the model what to do → run that one real tool → record it → repeat, up to 4
  steps. No LangChain, no AutoGen. That's a deliberate trade — less flexible, but every step is
  inspectable and debuggable, which matters a lot more than flexibility in a compliance context.
- **Counterfactuals are real model re-runs, not guesses.** When Chitrita asks "what if the amount
  were lower," the system doesn't ask an LLM to imagine the answer — it literally changes the
  number in the feature vector and re-runs the trained model, then reports the new score and
  which features moved.
- **Honest disclosure of synthetic vs. real data, in the UI itself, not a footnote.** The 10
  bank accounts in the real dataset are genuinely independent — there's no real laundering
  network in them. Rather than fake cross-account links to make a richer demo, the team built
  a *separate*, clearly-labeled synthetic network (with FATF-typology patterns seeded in on
  purpose) for the multi-account graph-walking story, and the UI tags every surface "real
  ledger" vs. "synthetic network" so nobody can mistake one for the other.
- **A model failing over gracefully is treated as a feature, not swept under the rug.** If a
  question takes more than 20 seconds or doesn't match anything, the system falls back to a
  pre-verified answer — and the UI says *"Using a prepared benchmark answer for this query"*
  out loud, rather than silently stalling or making something up.
- **Slightly technical but worth knowing if asked:** the fraud model isn't naive about class
  imbalance. Only 0.17% of the 284,807 transactions in the training data are fraud. The team
  ran a real bake-off between SMOTE (synthetic oversampling) and simple class-weighting before
  picking a calibrated SMOTE + LightGBM combination — see §8 for the numbers.

---

## 6. What actually works right now vs. what's partial/mocked — be blunt about this

**Fully working, verified by actually running it, safe to demo live:**
- All four backend services start, pass health checks, and serve real data from the real
  datasets (not sample/fake data).
- Card-fraud case: `get_transaction` → SHAP explanation → agent narrative → counterfactual
  re-scoring — all end-to-end real, hitting the real trained model.
- Ledger engine: real 10-account bank data (116,201 transactions), real anomaly detection,
  real per-account timelines.
- Typology engine: real generated synthetic network (29 accounts, 192 transactions, 36
  detected FATF flags), real graph-walk traversal.
- The chat/counterfactual panel, including the disclosed cached-fallback behavior.
- Automated test suite: 56/56 passing (see §8 for breakdown).
- A one-command startup script and a real end-to-end smoke test that starts all four services
  against real data and checks every critical path.

**Partial or worth knowing about before a judge digs in:**
- **The flagship "$4,850 off-hours" card-fraud story is the *mock-mode* narrative, not what
  live mode shows for that same ID — verified by hitting both directly.** `TX-CARD-9842` is
  treated as a literal row index into the real 284,807-row dataset. In **live mode** (the
  default once you `cp .env.example .env`, per `GETTING_STARTED.md`), row 9842 is a real,
  legitimate ~$8.69 transaction — the live fraud engine correctly scores it **0.0001, verdict
  "clear"**, not flagged at all. The "$4,850, 0.89 risk, flagged" story only appears in
  **mock mode** (`VERITY_ENV` unset/`mock`), where it's a hand-written fixture, not a live
  model inference. Neither number is wrong for its mode — but they are not interchangeable,
  and presenting the mock number while live services are visibly running would be misleading.
  **Recommendation: for a live-mode demo, use `TX-CARD-623` instead** — a real,
  fraud-labeled row (verified: live risk score 0.93, "flagged"; dropping the amount to $5
  moves it to 0.88, still flagged but visibly lower). It's less rehearsed than "$4,850 at
  3:22 AM" but every number on screen is then genuinely live. If you'd rather keep the
  polished $4,850 narrative, run in mock mode instead and say so on stage — see §7.
- **The mock-mode counterfactual path also disagrees with the mock-mode SHAP path for the
  same transaction — a real inconsistency, not just a live/mock difference.** In mock mode,
  `get_shap_explanation("TX-CARD-9842")` returns risk `0.89`, but
  `counterfactual("TX-CARD-9842", ...)` computes its *own* baseline via a separate small
  logistic-regression model (`agent/model_engine.py`) and reports `original_risk_score: 1.0`
  for the exact same transaction. Two different code paths, two different numbers, for the
  same ID. Not a live/mock issue — a genuine loose end between Person A's fraud-engine
  fixture and Person C's counterfactual engine that was never reconciled. Avoid showing both
  panels back-to-back for the same case in mock mode, or a sharp judge will ask why the
  numbers don't match.
- **The demo case IDs don't all line up with the live datasets, for the other two tiers.**
  The fixture cases you'll click through in the UI (`ACC-1092`, `TX-SYNTH-5501`, etc.) were
  invented early on as illustrative examples and don't match the real generated ledger/
  synthetic-network IDs. Clicking those cases correctly and honestly falls back to a saved
  fixture answer rather than fabricate a match — it doesn't crash, it just isn't pulling that
  *specific* number live. Treat the ledger/synthetic evidence panels as "here's what real
  live data from the same engines looks like" using whichever real account happens to load,
  rather than promising the exact pre-written numbers on screen.
- **The "AI" in the agent is a deterministic rules engine by default, not a live LLM call.**
  Unless someone configures an OpenAI-compatible API key, `agent/llm.py` uses a built-in,
  hand-coded decision engine instead of calling out to GPT/Claude/etc. This isn't a bug — it's
  documented as a deliberate mock/live toggle — but if a judge asks "is that a real LLM," the
  honest answer is "the tool-calling and reasoning steps are deterministic Python logic by
  default; an external LLM can be plugged in, but wasn't required to prove the architecture."
- **The fraud engine's API key currently ships inside the browser bundle.** Fine for an
  internal prototype demo; explicitly flagged as not production-ready (a real deployment would
  proxy that call through a server instead of the browser).
- **Nothing here is production-audited compliance software.** It's a decision-support
  prototype. Say this proactively — it's in the team's own pitch notes already
  (`docs/PITCH.md`) and it preempts the "would a regulator actually sign off on this" question.

**Not built / explicitly out of scope (say so if asked, don't dodge):**
- No user authentication/login system — this is a single-analyst prototype.
- No persistent database — case state lives in server memory and resets when the services
  restart.
- No production deployment story (Docker/cloud) — it's a local dev setup, documented as such.

---

## 7. Demo script

**Pick one mode before you start, and know why:**
- **Mock mode** (`VERITY_ENV` unset, or `mock`) — the polished "$4,850 off-hours, 0.89 risk"
  narrative on the card-fraud case is internally consistent (title, risk badge, and SHAP
  factors all agree), because it's all served from a hand-written fixture rather than live
  model inference. Safer for a rehearsed walkthrough; say plainly that this run is using
  fixture data for the narrative case while the engines behind it are real and running.
- **Live mode** (`VERITY_ENV=live`, the default once you `cp .env.example .env`) — every
  number on screen comes from the live trained model. The trade-off: the card-fraud case's
  static title/summary text ("$4,850 authorization... flagged") is a fixed UI label that does
  **not** update, while the risk-score badge and SHAP factors panel *do* fetch live — and for
  `TX-CARD-9842` specifically, the live numbers say **0.0001, clear**, not flagged (verified
  in §6). If you demo live mode, don't say "$4,850" out loud for that case; let the live
  panels speak for themselves, or use `TX-CARD-623` instead (verified live: 0.93, flagged).

**Recommendation for a pitch: mock mode for the rehearsed walkthrough below, live mode
available as backup proof if a judge asks "is this actually real" (show the `/health` metrics
or run `smoke_test.py` on request).**

### Setup (do this before judges arrive, not live)
```sh
git clone <repo-url> && cd Verity
git lfs pull                          # pulls the real datasets
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cd dashboard && npm install && cd ..
cp .env.example .env
# For the rehearsed walkthrough, comment out VERITY_ENV=live in .env (or set VERITY_ENV=mock)
# so the card-fraud case's numbers stay internally consistent — see note above.
python scripts/dev_up.py              # trains the fraud model on first run (~90s), then starts everything
```
Wait for `Engines online (4/4)` in the dashboard masthead before judges walk up. Leave this
running for the whole demo session — don't restart it between judges if you can avoid it (the
very first request to some endpoints is slower — see the timing note below).

### Live walkthrough (aim for ~4 minutes)
1. **Open http://localhost:3000.** Point at "Engines online (4/4)" — say: *"this isn't a
   screenshot, these are four live backend services."*
2. **Click the card-fraud case** (`CASE-CARD-001`, $4,850, off-hours). Point at the risk score,
   then the risk-factors panel — call out the split between "Amount"/"Time" (human-readable)
   vs. "V14" (anonymized signal, honestly labeled as not human-interpretable).
3. **Point at the reasoning trace panel.** Say: *"every line here is generated from a logged
   tool call, timestamped, with an event ID — the AI is not allowed to say anything that isn't
   in this list."*
4. **Type a counterfactual**: `What if the amount were $150 instead?` — in mock mode this
   reliably drops the score from a flagged baseline to "clear" (verified: 1.00 → 0.34 in this
   mode — see §6 for why mock mode's counterfactual baseline differs from its own SHAP number;
   don't state the "before" number out loud, just show the drop and verdict flip). In live
   mode, use `TX-CARD-623` for this step instead (verified: 0.93 → 0.88, stays flagged but
   visibly lower — a more honest "partial mitigation" story rather than a full clear).
5. **Switch to the ledger or synthetic case** to show the second evidence mode (timeline vs.
   network graph) — frame this as "the same architecture, two different data shapes," and if
   in live mode, be upfront that these two cases fall back to saved fixtures rather than the
   exact live account shown (§6).
6. **Ask an off-script question** to show the honest fallback: something not about this case,
   e.g. `What is the capital of France?` should get a graceful "no prepared answer" response
   rather than a made-up one; or ask one of the four benchmark questions (see fallback list
   below) to show the disclosed "using a prepared benchmark answer" badge.

### Backup plan if live demo breaks
- **If a service crashes or won't start:** you already ran `python scripts/smoke_test.py`
  before the pitch and have its full PASS output saved/screenshotted — show that as proof
  every path works, then fall back to a walkthrough of the UI with cached answers (the fallback
  Q&A below still work with services down, since they're local JSON).
- **If the model wasn't pre-trained:** `python scripts/dev_up.py` auto-trains it, but that's a
  ~90-second dead-air risk — never let this happen live, always pre-warm.
- **Four guaranteed-safe questions** (pre-verified, cached, work even if a live call times out):
  1. *"Why was this flagged?"*
  2. *"What if the amount were different?"*
  3. *"Show me a similar case."*
  4. *"Why wasn't this other account flagged?"*
- **If the dashboard itself won't load:** fall back to `curl`-ing the APIs directly on stage —
  all four `/health` endpoints, then (in live mode) `GET /api/v1/fraud/explain/TX-CARD-623`
  with the API key, which reliably returns a flagged, high-risk real transaction — less
  polished, but proves the engines are real, which is the actual claim.

---

## 8. Numbers you can cite (with sources)

| Metric | Value | Source |
|---|---|---|
| Card transactions in fraud dataset | 284,807 | `data/raw/creditcard.csv` (measured) |
| Fraud rate in that dataset | 0.172% (492 transactions) | Standard dataset stat, confirmed in `docs/DATASETS.md` |
| Currently trained model — precision | 90.5% | Live `GET /health` on the fraud engine (`v1.2-...` model) |
| Currently trained model — recall | 76.0% | Same |
| Currently trained model — F1 | 0.826 | Same |
| Currently trained model — PR-AUC | 0.805 | Same |
| Currently trained model — ROC-AUC | 0.984 | Same |
| Inference approach | LightGBM (gradient-boosted trees) + SHAP TreeExplainer, calibrated via 5-fold cross-validation with a time-based train/test split | `engines/fraud/train.py` |
| Bank ledger transactions | 116,201, across 10 real accounts | Measured by parsing `data/raw/bank.xlsx` |
| Ledger anomalies detected | 176 (balance breaks + timing spikes + reversal outliers, across all 10 accounts) | Live `GET /api/v1/ledger/anomalies` |
| Synthetic laundering network | 29 accounts, 192 transactions | Live `GET /api/v1/typology/network` |
| FATF typology flags detected in that network | 36 | Live `GET /api/v1/typology/flags` |
| Typology detector accuracy on held-out adversarial test set | 4/4 (100%) — includes 2 deliberately benign look-alikes designed to trigger false positives, and both were correctly *not* flagged | Live `GET /api/v1/typology/evaluation` |
| Automated test suite | 56/56 passing | `pytest -v`, run this session |
| Agent investigation loop | up to 4 reasoning steps per case, ~2-5 seconds typical | `agent/loop.py`, observed during smoke testing |
| Chat/counterfactual latency guard | falls back to a cached answer past 20 seconds, always disclosed | `agent/fallback.py` |
| Example flagged transaction (**live mode**) | `TX-CARD-623`: risk 0.93 → 0.88 after cutting the amount to $5 (stays flagged) | Live `GET /api/v1/fraud/explain/` + `POST /api/v1/agent/counterfactual`, verified this session |
| Example flagged transaction (**mock-mode narrative only**) | `TX-CARD-9842`: "$4,850, 0.89 risk, flagged" | Fixture in `contracts/mock_data/`; **do not cite as a live number** — live mode scores this same ID at 0.0001/"clear" (see §6) |

> **Caveat on the model numbers:** `docs/DATASETS.md` quotes a slightly different set of
> metrics (87.1% precision / 82.65% recall) from an earlier training run. The table above is
> the model **currently trained and running** in this checkout, pulled live from the running
> service, not from a stale doc — use these numbers, they're what a judge would see if they
> hit `/health` themselves. Both runs used the same method; small differences run-to-run are
> normal for a stochastic training process (SMOTE resampling + gradient boosting), not a
> methodology change.

---

## 9. Likely judge questions — short, honest answers

**"How does this scale to millions of transactions?"**
The fraud model inference is sub-millisecond per transaction, and LightGBM/SHAP scale well
horizontally. The current bottleneck isn't the model, it's the ledger engine's full-dataset
pandas parse on first request (no database yet — everything's in-memory/file-based). A real
deployment would need a proper database and pre-computed feature caching; this is a prototype
architecture, not a production one, and that's an explicit, known next step, not something
missed.

**"Is this secure enough for financial data?"**
No, not as-is, and we won't claim otherwise. There's no user auth, no encryption-at-rest
story, no audit log beyond the in-session trace, and (see §6) the fraud engine's API key is
currently embedded in the browser bundle rather than proxied server-side. It's a decision-support
prototype demonstrating an architecture, not a compliance-audited product.

**"Why not use an existing fraud/AML tool?"**
This isn't claiming to be the first agentic fraud tool — the pitch is specifically about
*trustworthiness*: most AI-assisted tools either black-box the score or let an LLM narrate
freely (hallucination risk). Verity's differentiator is the code-level grounding filter that
makes hallucinated claims structurally impossible to reach the UI, not a novel detection
algorithm.

**"Why no LangChain/AutoGen/agent framework?"**
Deliberate choice, documented from day one. A hand-rolled loop means every step is a plain
Python function call you can read, log, and test — no hidden retries, no framework-injected
prompt bloat, no version-to-version breakage risk. Trade-off: less flexible for arbitrary new
tools, but this system only ever needs exactly four tools, so the flexibility wasn't worth the
opacity.

**"What's mocked vs real right now?"** — see §6, answer it directly, don't spin it.

**"Why two separate engines (fraud vs. ledger) instead of one model?"**
Different data shapes and different failure modes. Card fraud is point-in-time transaction
scoring on a labeled dataset (supervised ML fits). Ledger/AML risk is about *patterns over
time* on unlabeled real accounts (statistical anomaly detection fits better, and there's no
ground truth to train a classifier against). Forcing them into one model would either overfit
the labeled fraud data or underuse the ledger's real structure.

**"What's next if you had more time?"** — see §11.

---

## 10. Who built what (so each teammate can speak to their piece)

Based on `docs/TEAM_ROLES.md` and each folder's actual commit history:

| Person | Owns | Can speak to |
|---|---|---|
| **Person A** | `engines/fraud/` — `benchmark_imbalance.py`, `train.py`, `explain.py`, `api.py` | Why SMOTE + class-weighting were both tried and compared; how SHAP splits interpretable vs. anonymized features; the calibrated-threshold approach; the fail-closed API-key design |
| **Person B** | `engines/ledger/` + `engines/typology/` + `data/synthetic/generate_network.py` | Parsing messy real bank narration strings into structured data; why the 10 real accounts are treated as independent (no fabricated cross-account links); how the synthetic network was seeded with real FATF typology patterns; the adversarial hold-out test set built specifically to catch false positives |
| **Person C** | `agent/` — `tools.py`, `loop.py`, `grounding.py`, `fallback.py`, `llm.py`, `model_engine.py`, `api.py` | The hand-rolled (no-framework) investigation loop; the code-level grounding/anti-hallucination filter and its offshore-account test case; the model-backed counterfactual engine; the 20-second latency watchdog and disclosed fallback design |
| **Person D** | `dashboard/` | The unified case queue UI, the tabbed real-ledger-timeline vs. synthetic-network views, the live/demo-exhibit labeling |

**One thing worth being upfront about if asked:** the four pieces above were built in parallel
against a shared contract (`contracts/schemas.json`) but weren't actually wired together and
tested end-to-end until an integration pass after the fact — that pass found and fixed several
real cross-team bugs (an auth header nobody was sending, a routing bug that silently sent
ledger/synthetic lookups to the wrong engine, a missing frontend folder that had never been
committed). Full details in `ARCHITECTURE.md` if a judge wants the receipts — it's a good
story about rigor, not something to hide.

---

## 11. Future scope — realistic next steps

- **Fix the demo-data mismatch** (§6): rebuild the illustrative demo cases around IDs that
  actually exist in the live datasets, so every case in the queue is fully live end-to-end, not
  just the card-fraud one.
- **Reconcile the two mock-mode risk numbers** (§6): make `agent/tools.py`'s mock counterfactual
  baseline read from the same fraud-engine fixture as `get_shap_explanation` instead of an
  independent mini-model, so a case's SHAP score and counterfactual "before" score always agree.
- **Real persistence.** Move case state out of server memory into an actual database so
  investigations survive a restart and analysts can review history.
- **User accounts and audit logging.** Required for any real compliance use — who looked at
  what, when, and what decision they made.
- **Wire in a real external LLM** for the reasoning/chat layer (the interface already supports
  this via `VERITY_LLM_API_KEY`) and compare its narrative quality against the deterministic
  built-in reasoner, still behind the same grounding filter.
- **Production auth for the fraud engine's API key** — proxy it server-side instead of shipping
  it to the browser.
- **Scale-test the ledger engine** against a much larger account set with a real database
  instead of an in-memory pandas parse.
- **Expand the FATF typology library** beyond the three currently implemented (structuring,
  round-tripping, rapid layering) — there are more recognized patterns in FATF guidance not yet
  covered.

#The presentation was completed on 19 September, 2026
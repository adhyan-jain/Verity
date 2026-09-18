# Feasibility Assessment — Agent-Drafted Suspicious Transaction Report (STR)

**Scope of this document:** a go/no-go assessment only. No code was written or changed to produce
this — everything below is grounded in reading `agent/grounding.py`, `agent/loop.py`,
`agent/llm.py`, `contracts/schemas.json`, `engines/ledger/anomalies.py`,
`engines/typology/fatf_rules.py`, and the official FIU-IND banking-company STR form
(fetched directly from fiuindia.gov.in — cited throughout).

---

## 1. Feasibility in this codebase

### 1.1 How `agent/grounding.py` actually works today

The filter is a **retain/reject gate per sentence**, not a citation-attachment system, in its
production path (`ground_narrative()`):

1. Splits a candidate narrative into sentences.
2. For each sentence, runs `is_sentence_strictly_grounded()`, which checks (in order): exact match
   against an approved trace-event sentence → unsupported entity IDs (`TX-`/`ACC-`/`EVT-`/etc.)
   → unsupported numbers → unsupported speculative terms (the "offshore account" test) → excess
   unsupported content words (allows at most 1 unsupported informative word).
3. Sentences that fail any check are **dropped silently**. `ground_narrative()`'s return value is
   just the surviving sentences joined into one string, plus the list of trace events used —
   **it does not return which sentence came from which event.**

**The good news:** the per-sentence citation mechanism this feature needs already exists, just
not wired into the production path. `find_supporting_event_id()` (token-overlap match between a
sentence and each trace event's `narration_sentence` + `tool_output_summary` + `tool_input`) and
`audit_grounding()` (which calls it and returns a `supporting_event_ids: {sentence: event_id}`
map for every retained sentence) are both already implemented and used today only for
test/debug auditing, not exposed through the `Case` object the UI consumes.

**What has to change:**
- `ground_narrative()` (or a new function) needs to return per-sentence citations as a first-class
  result, not require a second call to `audit_grounding()` bolted on afterward.
- The matching itself needs to get **stricter** for this use case. Two things in the current
  checks are looser than a document styled after a regulatory filing should tolerate:
  - Numeric matching uses substring containment (`num_clean not in evidence_corpus_clean`), not
    exact-value matching against the *specific* cited event's structured fields.
  - Entity-ID matching (`ident in ev_id`) is also substring/partial, not exact.
  These are reasonable for a chat-narrative filter (where "close enough, drop if truly novel" is
  the goal) but not rigorous enough as the *only* check backing a compliance-adjacent artifact —
  which is exactly the argument for building item 1.4 below rather than reusing this as-is.

### 1.2 Do the tool outputs carry enough to fill an STR?

I mapped this against the **actual official form** (FIU-IND, "Suspicious Transaction Report (STR)
for a Banking Company," fetched from `fiuindia.gov.in/pdfs/downloads/SBA.pdf`), not a guess at
what an STR "probably" contains:

| STR section | Real content required | Can Verity's tool outputs fill it? |
|---|---|---|
| Part 1 — Report details | Date sent, replacement flag | Trivial (today's date) — not data-derived |
| Part 2 — Principal Officer | Name, designation, bank identity, contact | **No.** Institutional/compliance-team metadata, unrelated to any case data. Would be static config, not agent output |
| Part 3 — Reporting Branch | Branch name, BSR code, address | **No.** Same as above |
| Part 4 — Individuals linked (+ Annexure A) | Customer name, ID, address, etc. | **No — empty on this data.** `creditcard.csv` is fully anonymized PCA data with zero PII by design. `bank.xlsx`'s parsed schema (`engines/ledger/parse_narrations.py`) has `account_id`, `amount`, `direction`, `balance`, `narration`, `payment_rail`, `counterparty` — the last is a regex-extracted guess from free-text narration, not a validated KYC name field. There is no customer-identity layer anywhere in this system |
| Part 5 — Legal persons/entities (+ Annexure B) | Entity name, registration, related individuals | **No — same problem.** The *synthetic* network does have invented `account_name` strings (e.g. "Apex Micro Holdings Ltd") — these are fabricated by a seeded generator and must never be presented as if real |
| Part 6 — Accounts linked (+ Annexure C) | Account numbers | **Partial.** Real ledger tier has real bank account numbers (`account_id` in `ReconciliationAnomaly`/`GraphWalkStep`). Card tier's `account_id` is nullable and typically null. Synthetic tier has account IDs that are real *within the synthetic graph* but not real accounts |
| Part 7.1 — Reasons for suspicion (checkboxes: identity of client / background of client / multiple accounts / activity in account / nature of transaction / value of transaction / other) | Category selection | **Mostly yes, by mapping.** `ReconciliationAnomaly.anomaly_type` → "activity in account"; `TypologyFlag.typology` (structuring/round-tripping/rapid-layering) → "nature of transaction" and arguably "multiple accounts"; high severity/confidence → "value of transaction." "Identity of client" and "background of client" **cannot** be filled — no KYC data exists |
| Part 7.2/7.3 — Grounds of Suspicion ("summary of suspicion and sequence of events") | Free-text narrative | **Yes — this is the one section Verity's engine genuinely has real material for.** `trace_events` already *are* a timestamped sequence of events; `ReconciliationAnomaly` carries severity, baseline/observed values, and evidence transaction IDs; `TypologyFlag` carries an FATF citation and confidence. This is the section worth building |
| Part 8 — Action taken | Investigation/agency status | **No.** Not knowable by this system; would need a human compliance-team input, not agent output |

**Bottom line: of 8 parts + 2 annexure types, exactly one (Part 7) is genuinely fillable from real
computed data, and one more (Part 6) is partially fillable for the ledger tier only.** Everything
else is either static configuration or must be explicitly left blank — never fabricated.

### 1.3 Deterministic rules engine vs. a real LLM

**Works today with the deterministic engine, no LLM required for the minimal version.** The
built-in reasoner (`agent/llm.py::_builtin_reasoning`) already produces sentences like:

> "Account history analysis for ACC-X detected a severe anomaly: balance plummeted into negative
> overdraft ($-3,200.00) and high-velocity burst of 4 consecutive transfers."

That's stiff, but it is already precise, factual, and close to the register FIU's own "grounds of
suspicion" instructions expect ("summary of suspicion and sequence of events") — the underlying
numbers (severity, z-score-derived risk, dollar amounts, timestamps) are real, computed values,
not invented. A real LLM's role here would be **purely stylistic** — turning already-correct,
already-grounded template sentences into smoother prose — not adding new facts. That also means
it adds hallucination surface without adding real value to the parts that matter (Part 7.1/7.2),
and it reintroduces the exact live-latency/availability risk the project already had to build a
fallback-and-disclose pattern for elsewhere (`agent/fallback.py`). **Recommendation: ship the
minimal version fully deterministic; treat an LLM prose pass as an optional, harder-gated
enhancement, never a dependency.**

### 1.4 What the validator should check, and can it be deterministic?

Yes, and it should be **deterministic-first**, not a second LLM call. Everything a validator needs
to check here is already structured JSON, not open-ended reasoning:

- **Numbers/amounts** — exact match (not substring) against the *specific* cited event's
  `raw_output` fields, not the whole trace corpus.
- **Dates/timestamps** — same, extending the existing regex extraction to ISO date patterns and
  checking against the cited event's `timestamp`/`window_start`/`window_end`.
- **Account/transaction IDs** — exact match (not partial) against the cited event's own IDs.
- **Typology claims** ("round-tripping," "structuring") — controlled-vocabulary check against
  `TypologyFlag.typology`'s enum value actually present in the cited event, not a fuzzy word match.

An "agent-as-a-judge" LLM call to re-check facts that are already exact structured data would be
strictly less reliable and non-reproducible compared to string/regex comparison. A second LLM
opinion is a reasonable idea for genuinely open-ended judgment calls — it is not the right tool
for "does this number match this JSON field." **Recommendation: build the validator as hardened,
per-citation-exact-match code (a stricter sibling of `is_sentence_strictly_grounded`, not a reuse
of it), and keep it deterministic even in the polished version.**

### 1.5 Files that would change / new files needed

- `agent/grounding.py` — extend to return per-sentence `(sentence, event_id)` pairs as a first-class
  result; tighten numeric/date/ID matching from substring to exact where the cited event is known.
- **New** `agent/str_draft.py` — maps a completed `Case` + its `trace_events` into an STR-shaped
  structure: Part 6 accounts (ledger tier only), Part 7.1 suggested checkboxes, Part 7.2 narrative
  with per-sentence citations, Parts 4/5 explicitly rendered as unavailable.
- **New** `agent/str_validator.py` — the hardened, independent second-pass check described in §1.4.
- `agent/api.py` — new endpoint, e.g. `POST /api/v1/agent/str-draft`.
- `contracts/schemas.json` — new `STRDraft` definition, following this project's existing
  contract-first convention.
- **New** `tests/test_str_draft.py`, `tests/test_str_validator.py` — minimum coverage: every
  retained sentence has a citation; the validator rejects a deliberately tampered draft (e.g. a
  number changed after grounding); a card-tier case correctly leaves Parts 4/5/6 empty rather than
  inventing content.
- **Dashboard**: one new view/panel (same visual language as the existing reasoning-trace panel) —
  a "Draft STR" action on a case, a review screen with citation chips per sentence (clicking one
  highlights the matching trace event, reusing the existing trace panel), and an "Approve draft"
  action that records a local audit entry — never a "submit" or "file" action, since no filing
  integration exists or should exist in a prototype.

### 1.6 Effort estimate

**Minimal version** (deterministic only, ledger tier, no new LLM dependency):
- Backend: citation attachment in grounding.py (2–3h) + section-mapping module (4–6h, mostly
  formatting since the underlying data already exists) + deterministic validator (3–4h) + new
  endpoint (1h) + schema (1h) + tests (3–4h).
- Frontend: draft view + citation chips + approve action (4–6h), reusing existing component patterns.
- **Total: ~18–25 hours.** Falls naturally to whoever owns `agent/` (extends their existing
  module directly) plus dashboard for the UI. Requires zero changes to the fraud/ledger/typology
  engines — pure consumption of what they already output, so it doesn't compete for the same
  people's time as other work.

**Polished version** (adds optional gated LLM prose, print/PDF layout resembling the real form,
richer checkbox suggestions, explicit empty-state design, adversarial test coverage):
- Additional ~15–20 hours on top of minimal.
- **Total: ~35–45 hours.**

---

## 2. Is it meaningful?

### 2.1 What an FIU-IND STR actually contains (cited)

Source: FIU-IND, *Suspicious Transaction Report (STR) for a Banking Company*, official form PDF —
`https://fiuindia.gov.in/pdfs/downloads/SBA.pdf` (fetched directly; all section numbers and
checkbox categories below are transcribed from that document, not inferred).

- **Part 1**: report metadata (date, replacement-report flag).
- **Part 2 / Part 3**: Principal Officer and reporting branch details — bank-side identity, not
  transactional.
- **Part 4 / Annexure A**: individuals linked to the transaction (name, customer ID if allotted,
  address) — one annexure per individual.
- **Part 5 / Annexure B**: legal persons/entities linked, including related individuals
  (director/partner/member) — one annexure per entity.
- **Part 6 / Annexure C**: accounts linked to the transaction, one annexure per account.
- **Part 7.1 — Reasons for suspicion** (multiple selection, from the form's own instructions):
  **A** Identity of client, **B** Background of client, **C** Multiple accounts, **D** Activity in
  account, **E** Nature of transaction, **F** Value of transaction, **Z** Other. The instructions
  give worked examples per category — e.g. under "D," "unusual activity compared with past
  transactions" and "sudden activity in dormant accounts"; under "E," "nature of transactions
  inconsistent with what would be expected from declared business"; under "F," "value just under
  the reporting threshold amount in an apparent attempt to avoid reporting."
- **Part 7.2/7.3 — Grounds of Suspicion**: free-text, explicitly instructed to be a "summary of
  suspicion and sequence of events."
- **Part 8 — Details of action taken**: whether the matter is/was under investigation by another
  agency, with contact details.

The general instructions also state the PMLA 2002 definition actually being reported against: a
transaction "gives rise to a reasonable ground of suspicion that it may involve the proceeds of
crime," or "appears to be made in circumstances of unusual or unjustified complexity," or "appears
to have no economic rationale or bonafide purpose." I did not find a separate, distinct
consolidated "STR narrative template" beyond this form on fiuindia.gov.in in this research pass —
everything cited above is transcribed directly from the SBA.pdf form and its own instructions
page; I have not verified whether a materially different or updated version exists post-2014
filing-structure changes, and that should be checked before anything resembling this ships beyond
a demo.

### 2.2 Does this strengthen the grounding claim, or is it cosmetic?

**Genuinely strengthens it, conditionally.** It's not bolted-on scope — it's the same grounding
mechanism the whole project is built around, applied to a document type with real external
structure and real stakes, which makes the claim more concrete and demonstrable, not just
restated. The condition: it only strengthens the claim if (a) citations are real, per-sentence,
and independently re-validated (not just the existing chat narrative reformatted into boxes), and
(b) sections with no real data are explicitly marked unavailable rather than silently skipped or
invented. Skip either of those and it becomes cosmetic — worse, it becomes a credibility risk,
covered next.

### 2.3 Would the draft be substantive, or thin? Blunt answer, per tier:

- **Card-fraud tier: thin, and arguably the wrong document type entirely.** SHAP factors
  (interpretable Amount/Time + anonymized V-signals) aren't "grounds of suspicion" in the FIU
  sense — an STR is fundamentally an AML instrument tied to a customer relationship and behavior
  pattern, not a point-in-time fraud-score decline. Forcing every tier through the same "draft an
  STR" flow is the single weakest part of this idea as stated. **This is a real design question,
  not just an implementation detail** — worth deciding explicitly whether the card tier gets a
  differently-named/differently-shaped output (e.g. a fraud investigation summary) or no draft
  action at all.
- **Real ledger tier: the legitimate case.** Real account numbers, real anomaly windows, real
  severity scores, real evidence transaction IDs. Parts 4/5 (identity) still correctly stay empty
  — and that's fine, as long as it's *shown* as fine.
  - **Empirical caveat on how much material this really gives you, from a live check this
    session:** ledger anomaly detection has no ground-truth labels at all (nothing in this data
    says an account is actually laundering money), so "grounds of suspicion" here means "this
    account is a statistical outlier against its own 3-month baseline" — real, computed, honest,
    but a materially weaker basis than what a bank's real KYC/behavioral history would give a
    human investigator. The draft should read as evidence-backed, not as proof.
- **Synthetic tier: the best-looking narrative, and the biggest risk.** FATF citations, multi-hop
  evidence, clean structuring/round-tripping stories — and **zero of it is real**. A
  polished-looking STR built on a fabricated network could look *more* convincing than the honest,
  messier real-ledger case while being *less* real — exactly backwards, and exactly the kind of
  inconsistency a judge with real compliance exposure would catch immediately. If this tier is
  included at all, the synthetic labeling needs to be at least as loud as everywhere else in the
  app already does it (the tabs are already labeled "real ledger" vs. "synthetic network" for
  exactly this reason).

### 2.4 Risks and how the UI should frame them

- **Fabricated-looking subject details** — the dominant risk. Never populate Parts 4/5 with
  synthetic `account_name` strings or narration-derived "counterparty" guesses as if they were
  verified identity. Render an explicit **"Not available — no KYC/customer-identity data in this
  system"** state instead of omitting the section silently. Silence reads as an oversight;
  an explicit non-claim reads as the same honesty the rest of the product already practices
  (see `docs/PRD.md` §6, "What We Do NOT Claim").
- **Tipping off** — not really operative for a prototype with no real customers and no filing
  channel, but the UI copy should still never imply a filing has occurred; real STR filings carry
  a statutory confidentiality obligation and the demo shouldn't casually borrow that language.
- **PII handling** — `bank.xlsx` is real (gitignored) data. If raw narration text (the regex
  "counterparty" extraction) is ever surfaced verbatim in a citation view without review, that's
  the closest thing to a real PII exposure anywhere in this system. Worth an explicit check before
  demoing.
- **Over-claiming regulatory readiness — the most important framing risk.** Recommend a
  **persistent, non-dismissable banner** on the draft view: *"Draft for analyst review only — not
  a filing, not submitted to FIU-IND, not legal advice."* Avoid the words "file"/"submit"/"filed"
  anywhere in this feature's UI; use "draft," "prepare," "review" only.

---

## 3. Verdict

### Go, with reduced scope.

**Why go:** the hard part — per-sentence citation infrastructure — already exists in this codebase
(`audit_grounding`/`find_supporting_event_id`), just not wired into the production path. This is a
real, cheap extension of tested, working code, not a new subsystem. It makes the project's actual
thesis (grounded, auditable AI in a compliance context) into a concrete artifact a judge can point
at, which is more memorable than another dashboard screen. And the research is unambiguous about
*where* the real value is: Part 7 of the actual FIU form maps directly onto data this system
already computes.

**Why reduced scope, specifically:**
1. **Ledger tier only for the demo/initial version.** It's the only tier with real, non-fabricated
   account-level evidence. Card-fraud tier is the wrong document type for an STR and should either
   get a differently-framed output or no draft action. Synthetic tier, if included at all, needs
   labeling as loud as the rest of the app already uses for it.
2. **Deterministic validator only.** No external LLM dependency for the meaningful version — it
   isn't needed for the one section (Part 7) that matters, and it removes a live-demo failure mode
   this project has already had to engineer around elsewhere.
3. **Parts 1–6/8 either static config or explicitly marked unavailable — never fabricated.** This
   is the difference between a feature that strengthens the grounding story and one that undermines it.

### Smallest version that's still meaningful

1. Extend `ground_narrative`/`audit_grounding` to return `(sentence, event_id)` pairs as a
   first-class result, with exact (not substring) matching for numbers/dates/IDs.
2. New endpoint mapping a completed ledger-tier `Case` into: Part 6 accounts, Part 7.1 suggested
   checkboxes (from `anomaly_type`/`typology`), Part 7.2 narrative with inline citations, Parts 4/5
   explicitly "unavailable."
3. A hardened, independent deterministic validator re-checking every cited fact against the
   *specific* event it claims to come from, run as its own visible pass — not silently reused from
   the chat grounding step, so it demonstrably is a second, separate check.
4. One dashboard view: sectioned draft, clickable citation chips that highlight the source trace
   event, persistent non-filing banner, "Approve draft (not filed)" action that logs to the
   existing decision/audit trail.

### 60-second demo path

Select the real-ledger case → trigger investigation (already works, ~5–10s) → click "Draft STR" →
sectioned draft renders → click a citation chip on a Grounds-of-Suspicion sentence, watch it
highlight the exact trace event it came from in the panel the judges already saw two minutes
earlier → point at Parts 4/5 explicitly marked unavailable ("we don't fabricate identity data") →
click "Approve draft (not filed)" → show it logged. This demo is strong specifically *because* it
reuses UI the judges have already seen — it reads as "the same trustworthy engine, one more output
format," not a bolted-on new feature.

### If you decide no-go instead

Not the recommendation here, but if the reduced-scope version still feels like too much surface
area to add before submission, the same ~20 hours would be better spent on the two gaps already
flagged as higher-leverage in the earlier discussion: closing the cross-dataset correlation gap
(the three data tiers currently never intersect) or strengthening the ledger-tier detection's
honesty story (making the "no ground truth exists" limitation an explicit, demoable part of the
pitch rather than a caveat).

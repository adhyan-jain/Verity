# Verity — Final Build Spec

This is the locked spec. No more architecture debate — this document plus the dataset files in `data/raw/` is everything needed to start building.

## 0. One-liner and persona (resolved)

**Persona:** Chitrita, a financial-crime analyst at a mid-size digital-first bank / NBFC. The bank is too small to run separate fraud and AML teams, so one combined "financial crime desk" handles both card-fraud alerts and account-risk investigation. This is the concrete, defensible answer to "who uses this" — state it as a named persona in the pitch, not as an abstract claim.

**One-liner:** Verity is Chitrita's single workspace — two detection engines feeding one case queue, with an AI agent that investigates each flagged case and can be questioned about its own reasoning, live.

## 1. Repo structure

```
Verity/
  data/
    raw/
      creditcard.csv          # provided
      bank.xlsx                # provided
    synthetic/
      generate_network.py      # builds FATF-typology-seeded network
      synthetic_network.json   # generated output
      adversarial_set.json     # held-out test cases, built independently of detect.py's author
  engines/
    fraud/
      benchmark_imbalance.py   # SMOTE vs class-weighting comparison — run first, pick a winner
      train.py
      model.pkl
      explain.py                # SHAP, splits interpretable (Time/Amount) vs anonymized (V1-V28)
      api.py                     # serves get_transaction, get_shap_explanation
    ledger/
      parse_narrations.py
      reconcile.py               # per-account baselines
      anomalies.py                # balance breaks, timing spikes, reversal outliers
      timeline.py                 # builds the per-account visual timeline (see section 5)
      api.py                       # serves walk_graph (real tier), reconciliation anomalies
    typology/
      fatf_rules.py                # structuring / round-tripping / rapid-layering, cited to FATF definitions
      detect.py
      api.py                        # serves walk_graph (synthetic tier), typology flags
  agent/
    tools.py     # thin wrappers around the three engines' APIs — the only way the agent touches data
    loop.py      # hand-rolled tool-calling loop, no framework
    grounding.py # enforces: every narrated sentence must cite a trace_event id
    fallback.py  # cached top-4 Q&A + "investigating" animation state (see section 6)
  contracts/
    schemas.json    # single source of truth — section 2 below
    mock_data/       # fixtures matching schemas, for Person D to build against before real APIs exist
  dashboard/
    case_list/
    graph_view/
    reasoning_trace_panel/
    chat_panel/
  docs/
    PITCH.md          # includes the "what we don't claim" slide
    DEMO_SCRIPT.md    # exact click-path + fallback answers
```

## 2. Data contracts (define these together, hour one, before anything else)

```json
// TransactionRecord
{
  "id": "string",
  "tier": "real_card | real_ledger | synthetic_network",
  "timestamp": "ISO8601",
  "account_id": "string | null",
  "amount": "number",
  "direction": "debit | credit | null",
  "raw_narration": "string | null",
  "source_dataset": "string"
}

// FraudExplanation
{
  "transaction_id": "string",
  "risk_score": "number 0-1",
  "verdict": "flagged | clear",
  "top_factors": [
    { "feature": "Amount", "human_label": "Transaction amount", "contribution": "number", "interpretable": true },
    { "feature": "V14", "human_label": "Anonymized signal V14", "contribution": "number", "interpretable": false }
  ],
  "model_version": "string"
}

// ReconciliationAnomaly (ledger engine, real tier)
{
  "account_id": "string",
  "anomaly_type": "balance_break | timing_spike | reversal_outlier",
  "window_start": "ISO8601",
  "window_end": "ISO8601",
  "severity": "number",
  "baseline_value": "number",
  "observed_value": "number",
  "evidence_transaction_ids": ["string"]
}

// GraphWalkStep (real ledger or synthetic network, tagged)
{
  "step_index": "number",
  "from_account": "string",
  "to_account": "string",
  "tier": "real_ledger | synthetic_network",
  "amount": "number",
  "timestamp": "ISO8601",
  "narration": "string | null",
  "tool_call_id": "string"
}

// TypologyFlag (synthetic tier only)
{
  "flag_id": "string",
  "typology": "structuring | round_tripping | rapid_layering",
  "fatf_reference": "string",
  "involved_accounts": ["string"],
  "evidence_transaction_ids": ["string"],
  "confidence": "number"
}

// AgentTraceEvent — the grounding unit. One per tool call.
{
  "event_id": "string",
  "case_id": "string",
  "timestamp": "ISO8601",
  "tool_called": "get_transaction | get_shap_explanation | walk_graph | counterfactual",
  "tool_input": {},
  "tool_output_summary": "string",
  "narration_sentence": "string"   // the ONLY sentence the agent is allowed to say for this event
}

// Case — top-level object the dashboard renders
{
  "case_id": "string",
  "tier_origin": "real_card | real_ledger | synthetic_network",
  "status": "open | investigating | closed",
  "primary_transaction_id": "string",
  "risk_score": "number",
  "trace_events": ["AgentTraceEvent"],
  "narrative": "string"   // built by concatenating trace_events[].narration_sentence, nothing else allowed
}
```

**Grounding rule, enforced in code (`agent/grounding.py`), not just prompted for:** the agent's `narrative` field can only ever be assembled from `narration_sentence` values that came from actual `AgentTraceEvent`s. If the agent's LLM call produces a sentence not backed by a trace event, `grounding.py` strips it before it reaches the dashboard. This is the direct fix for hallucinated hops — it's a code-level filter, not a prompt instruction.

## 3. Detection engines — what each one does and its one job

- **Fraud engine**: run `benchmark_imbalance.py` first (SMOTE vs. class-weighting on recall). Ship the winner. Output feeds `get_transaction` and `get_shap_explanation`.
- **Ledger engine**: per-account only — no cross-account network claims. Outputs `ReconciliationAnomaly` and real-tier `GraphWalkStep`s (a "walk" here just means the account's own transaction sequence, not a multi-party hop).
- **Typology engine**: synthetic-tier only. `fatf_rules.py` implements structuring/round-tripping/rapid-layering per FATF's actual definitions (cite them in code comments and in the pitch). `adversarial_set.json` must be built by a teammate who did not write `detect.py` — assign this explicitly, don't let it default to whoever's fastest.

## 4. Agent core

- `tools.py` exposes exactly four functions to the LLM: `get_transaction`, `get_shap_explanation`, `walk_graph`, `counterfactual`. Nothing else is callable.
- `loop.py` is a plain explicit loop (call model → get tool call → execute → append trace event → repeat until model returns a final narrative or hits a max-hop limit). No agent framework.
- Every hop must produce a trace event before the agent is allowed to reference it in a sentence (enforced by `grounding.py`).
- `counterfactual` re-runs the relevant engine with one input changed and returns a fresh `FraudExplanation` or `TypologyFlag` — it does not let the LLM just assert a hypothetical answer.

## 5. Ledger visual fix (resolves the sparse-data asymmetry)

Don't present the ledger surface as a short flag list next to the fraud surface's rich case queue. Each of the 10 accounts gets its own **timeline view**: a horizontal strip of that account's transactions over time, with `ReconciliationAnomaly` windows highlighted in place (a timing spike shows as a visibly denser cluster; a reversal outlier shows as a marked point). This gives the ledger surface real visual density even with only 10 accounts — `timeline.py` builds this data, `dashboard/graph_view/` renders it as a distinct mode from the synthetic-network graph, clearly tab-labeled "real ledger" vs. "synthetic network" so the tier distinction stays visible in the UI, not just in a footnote.

## 6. Chat panel failure mode (resolves the stall risk)

- Default state on any tool call: an "investigating…" panel that live-updates with each `AgentTraceEvent` as it lands (e.g. "called `walk_graph`… found 2 hops" appears within a couple seconds of the actual call, not held back until the whole chain finishes). This makes 15-30 seconds of real latency read as visible work.
- Pre-build cached, fully-real answers (not fabricated — actually run once and saved) for the four most likely judge questions: "why was this flagged," "what if the amount were different," "show me a similar case," "why wasn't this other account flagged." If a live call exceeds ~20 seconds, fall back to the cached answer for a matching question, but say so explicitly in the UI ("using a prepared answer for this one") — silent fallback would break the honesty principle the whole pitch rests on.

## 7. Build order (contract-first, integration pulled earlier)

| Time | Person A (fraud) | Person B (ledger + synthetic) | Person C (agent) | Person D (dashboard) |
|---|---|---|---|---|
| 9am start | **All four**: finalize `contracts/schemas.json` and `mock_data/` together (30-45 min) before splitting | | | |
| 9am-1pm | Run imbalance benchmark, train v1 model | Parse `bank.xlsx`, build v1 reconciliation anomalies + timeline data | Scaffold `loop.py` and `tools.py` against `mock_data` only | Build dashboard shell against `mock_data` |
| 2-5pm | Wire real fraud API + SHAP split | Finalize reconciliation anomalies; start `generate_network.py` with FATF typologies | **First real integration**: wire tools to A's real fraud API | Wire case list + explanation panel to A's real API; build timeline view |
| **5-8pm Review 1** | Demo: real fraud case, SHAP explanation, agent narrating one fraud case end-to-end, ledger timeline view | | | |
| 9pm-1am | Support/freeze fraud engine | Finish synthetic network + `adversarial_set.json` + `detect.py` | **Second real integration** (the harder one, done here not at 4am): wire `walk_graph`/`counterfactual` to real+synthetic APIs | Wire graph view (tabbed real/synthetic) + reasoning-trace panel |
| **1-4am Review 2** | Demo: full agent walking real ledger + synthetic network, live trace panel, chat Q&A | | | |
| 4-6am | Freeze | Tune false-positive rate | Bug-fix already-integrated pieces only — no first-time integration here | Build chat fallback state + cached answers |
| 8-10am | Metrics slide | Final data for dashboard | Prompt polish | Full rehearsal, 3+ run-throughs |
| 10am-1pm Final | Present: rigor, honesty, grounded agent, one analyst desk | | | |

## 8. What NOT to claim (say this out loud in the pitch)

- Not an interconnected laundering network in the real ledger — it's 10 largely independent accounts, disclosed as such.
- Not a reverse-engineering of anonymized PCA features into human meaning.
- Not the first agentic fraud/AML tool — the edge is execution rigor and honesty, not category novelty.
- Not a production-audited compliance system — a decision-support prototype.

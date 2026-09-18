# Pre-Run Judge Q&A Cache

The 4 most likely judge questions (matching the canonical categories in
`contracts/mock_data/mock_fallback_qa.json`), pre-run against the live
system this session on the demo script's chosen transaction (`TX-CARD-623`,
$529.00, live risk score 0.9286, flagged — see `docs/DEMO_SCRIPT.md`).

## 🚩 Urgent flag before using this in a demo

**The cached fallback answers for Q1 and Q2 in
`contracts/mock_data/mock_fallback_qa.json` still reference the old
`TX-CARD-9842` / $4,850 narrative** — the one `PITCH_BRIEF.md` already
flagged as scoring completely differently in live mode ($15.95, 0.0001 risk,
"clear"). If the live agent call fails during the demo and falls back to
that cached answer while `TX-CARD-623` ($529) is what's shown on screen,
**the audience will hear about a transaction that isn't the one they're
looking at.** This is outside `validation/`/`docs/`/`demo/` — I don't edit
`contracts/mock_data/` — flagging to whoever owns that fixture (Person C per
`docs/TEAM_ROLES.md`) to update it to match `TX-CARD-623` before this script
is used live.

## Q1 — "Why was this flagged?"

**Live answer, captured this session** (real agent investigation +
LLM-generated chat answer, Ollama qwen3:8b, `agent/llm.py::generate_chat_answer`):

> "The case was flagged due to an elevated risk score of 0.93 identified by
> the SHAP model, which is driven by the following factors: 1. Transaction
> amount: The $529.00 debit transaction on rail creditcard.csv contributed
> a +0.45 impact to the risk score. 2. Anonymized signals: Internal model
> features (V4, V14, V12) added cumulative weights of +3.00, +2.33, and
> +1.46, respectively, further increasing the risk score. The model's risk
> threshold likely triggered the flagging, indicating potential abnormal or
> high-risk behavior associated with the transaction."

**Source:** live LLM call. **Fallback if it doesn't fire live:** the cached
answer in `mock_fallback_qa.json` — currently mismatched (see flag above).

## Q2 — "What if the amount were different?"

**Live answer, captured this session** (real counterfactual re-run via
`engines/fraud/api.py::compute_counterfactual`, actual model inference, not
a lookup):

| Override | Recalculated risk | Verdict |
|---|---|---|
| $529.00 (original) | 0.9286 | flagged |
| $50.00 | 0.8755 | flagged |
| $10.00 | 0.8730 | flagged |
| $5.00 | 0.8773 | flagged |
| $1.00 | 0.7679 | flagged |

**This transaction's verdict does not flip even at $1** — the anonymized
behavioral signals (V4/V14/V12) dominate the risk score for this specific
case, not the dollar amount. This is worth saying explicitly if asked: it's
a more honest and more convincing answer than a scripted "watch it flip to
clear" demo, because it's what actually happened when the real model was
re-run, not a cherry-picked example.

**Source:** live model re-run, no LLM involved (the counterfactual endpoint
computes directly, matching "real model re-run, not a guess" from
`PITCH_BRIEF.md`). **Fallback if the endpoint is unreachable:** cached
answer in `mock_fallback_qa.json` — same mismatch flag as Q1 applies.

## Q3 — "Show me a similar case"

**No live capability exists for this today.** The agent has exactly 4 tools
(`get_transaction`, `get_shap_explanation`, `walk_graph`, `counterfactual`
— confirmed via `agent/llm.py`'s system prompt and `agent/tools.py`) — none
of them perform similarity search across historical cases. **This question
will always route to the cached fallback answer**, every time, not just
under latency pressure:

> "Found 3 historical cases matching the profile: CASE-CARD-082 (score 0.91,
> off-hours high-velocity purchase), CASE-CARD-044 (score 0.87, high amount
> burst), and CASE-CARD-119 (score 0.86, international IP jump)."

**Say this explicitly if asked live:** "similar-case search isn't a live
capability yet — that's a cached, illustrative answer, and the UI will show
the fallback notice when you ask it." Don't let this one pass as if it were
live; unlike Q1/Q2, it structurally never can be.

## Q4 — "Why wasn't this other account flagged?"

**No live capability exists for the card-fraud tier** (there's no "why not"
query path in `agent/tools.py`). The cached answer is written for the
**ledger tier** (a specific account, `ACC-1080`), not the card-fraud
transaction this demo path uses:

> "Account ACC-1080 was evaluated against the ledger baseline model:
> variance across velocity (0.4 std dev) and reversal rate (0.01) remained
> strictly within normal 95th-percentile operational bounds."

**This question only makes sense if asked during the Ledger Engine step**
(step 3 of the demo script), not during the fraud/SHAP step — steer a judge
who asks this toward the ledger tier example, or the fallback answer will
visibly reference an account that was never shown on screen.

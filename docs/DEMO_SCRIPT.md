# Verity — Demo Script

**Mode:** live (not mock). All numbers below were verified live in this
session against the actual running fraud engine, not asserted from a doc.
See `validation/claims_ledger.md` for the audit that produced this rewrite —
the previous version of this script didn't specify mock vs. live and used
`TX-CARD-9842` from the mock-mode narrative, which scores as a $15.95,
0.0001-risk "clear" transaction in live mode (confirmed live, this session)
— using it on stage in live mode would visibly contradict the "$4,850
off-hours fraud" story. **Use `TX-CARD-623` instead — verified below.**

**Total run time: ~4 minutes** if nothing needs a fallback cut; see the
failure-path table for exactly when to cut and how long that adds.

## Click path with timing

| Step | Time budget | Action | What to say |
|---|---|---|---|
| 1. Case Queue Overview | 0:00–0:30 | Show the unified case queue (masthead shows "Engines online (4/4)" — a real health check, not decoration) | "One analyst, one queue, three alert types instead of three separate tools." |
| 2. Fraud Detection & SHAP | 0:30–1:15 | Open **`TX-CARD-623`** (real amount $529.00, verified live risk score **0.9286, flagged**). Show the SHAP panel: top interpretable factor is `Amount` (contribution +0.4532); the dominant factors are anonymized `V4`/`V14`/`V12` (contributions +3.00/+2.33/+1.46) | "Interpretable factors get plain-English labels; V1–V28 are PCA-transformed and stay anonymized — we don't invent meaning for them." |
| 3. Ledger Engine & Timeline | 1:15–2:00 | Switch to Real Ledger tab, show one account's timeline with a balance-break/timing-spike/reversal-outlier highlighted | "This tier is single-account only — no cross-account network claims here, that's the synthetic tier next." |
| 4. Agent Investigation & Trace | 2:00–3:00 | Trigger an investigation on `TX-CARD-623`. Watch the trace stream populate step-by-step (`get_transaction` → `get_shap_explanation` → narrative). If the LLM path is live (Ollama/OpenRouter configured), narrate that; if it silently falls back mid-stream, **say so out loud** the moment the UI's fallback notice appears — don't let it pass silently. | "Every sentence in that narrative is grounded — code-level filtered against what the tools actually returned, not generated freehand." |
| 5. Counterfactual Q&A | 3:00–4:00 | Ask: **"What if the amount were $50 instead of $529?"** Verified live answer: risk drops from **0.9286 → 0.8755**, verdict **stays "flagged."** This is a better demo moment than a scripted flip — it shows the counterfactual is a real model re-run (the anonymized behavioral signals, not the dollar amount, are driving this particular transaction's risk), not a canned "watch it flip" trick. | "That's not us picking a nicer-looking number — the model genuinely doesn't change its mind here, because Amount isn't what's driving this one. That's the honest answer, and it's more convincing than a scripted flip would be." |

## Failure paths — when to cut to a fallback, and the disclosure rule

**The rule, non-negotiable:** if a fallback answer is shown, **the UI must say so explicitly.** This is already wired — confirmed in this session by reading `dashboard/src/components/verity-workspace.tsx:749-750`, which renders a visible notice (`"Using a prepared benchmark answer"` or the API's own `fallback_notice` string) whenever `is_fallback` is true or the live call throws. Silent fallback would directly contradict the pitch's own honesty principle — don't let anyone read a fallback answer aloud as if it were live without also reading the notice.

| Step | Cut-to-fallback trigger | What happens automatically | What you say |
|---|---|---|---|
| Step 4 (agent trace) | No visible progress in the trace stream for 5 seconds, or `agent/fallback.py::execute_with_latency_guard`'s timeout fires (default 20s) | Falls back to a cached Q&A answer (`contracts/mock_data/mock_fallback_qa.json`) automatically — this is already real code, not something you trigger manually | "That's the fallback path — same one the honesty principle requires we disclose, and it just did, right there in the notice." |
| Step 5 (counterfactual/chat) | Same latency guard; also triggers on any exception from the live call | UI shows `"Using a prepared benchmark answer · model not rerun"` per `verity-workspace.tsx:731` | Same as above — read the notice text out loud, don't just move past it |
| Any step | All 4 backend services fail the "Engines online (4/4)" health check before starting | **Do not start the live demo.** Switch to a full walkthrough of pre-recorded screenshots/video instead of live-clicking a broken system | State plainly: "We're running the recorded walkthrough because the live backend isn't healthy right now — here's what it looks like when it is" |

## Pre-run judge Q&A

See `demo/judge_qa_cache.md` for the 4 most likely judge questions, pre-run
against the live system this session, with actual verbatim answers captured
— not invented.

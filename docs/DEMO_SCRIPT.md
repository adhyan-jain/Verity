# Verity — Demo Script

## Demo Flow & Click-Path

1. **Case Queue Overview**
   - Show unified alert inbox feeding from both Fraud and Ledger detection engines.
   - Highlight Priya's workflow (one financial crime desk handling card fraud & account risk).

2. **Fraud Detection & SHAP Explanation**
   - Open flagged card-fraud alert.
   - Inspect SHAP factor breakdown: clear separation of interpretable factors (Time, Amount) and anonymized signals (V1-V28).

3. **Ledger Engine & Timeline View**
   - Switch to Real Ledger tab.
   - Show per-account transaction timeline with visual density: highlight balance breaks, timing spikes, and reversal outliers across accounts.

4. **Agent Investigation & Reasoning Trace**
   - Trigger AI Agent investigation.
   - Inspect `AgentTraceEvent` stream in real-time (investigating panel updates step-by-step).
   - Verify code-level grounding: narrative is strictly composed of verified trace sentences.

5. **Live Interactive Q&A / Counterfactuals**
   - Ask counterfactuals ("What if the amount were $50 instead of $5,000?").
   - Watch counterfactual engine re-compute score rather than hallucinating answers.

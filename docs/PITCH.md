# Verity — Pitch Guide

## 0. Persona & One-Liner
- **Persona:** Chitrita, a financial-crime analyst at a mid-size digital-first bank / NBFC.
- **One-Liner:** Verity is Chitrita's single workspace — two detection engines feeding one case queue, with an AI agent that investigates each flagged case and can be questioned about its own reasoning, live.

## 1. The Anomaly Taxonomy — Open With This

Most fraud/AML pitches show one classifier and call it done. Anomaly
detection actually spans four distinct categories, and no single classifier
covers all four well. Opening with this taxonomy — and being explicit about
which quadrant is covered, how well, and which isn't — is what separates a
research-grounded system from a weekend classifier demo.

| Category | What it means | Verity's coverage |
|---|---|---|
| **Point anomalies** | A single data point that is anomalous relative to the rest of the data (e.g. one transaction with an unusual feature profile) | **Covered.** The fraud engine's per-transaction SHAP-explained LightGBM classifier on `creditcard.csv`, independently validated metrics in `validation/claims_ledger.md`. |
| **Contextual anomalies** | A point that's anomalous only relative to its own context (e.g. this account's own baseline), not the global population | **Covered.** The ledger engine's per-account reconciliation (balance breaks, timing spikes, reversal outliers) — each account measured against its own history, not a global norm. |
| **Collective anomalies** | A group of points that's anomalous only as a group, even if no individual point looks unusual alone (e.g. a laundering ring) | **Covered, on a synthetic network.** The typology engine detects structuring, round-tripping, and rapid layering across a multi-hop network — independently validated for structuring specifically in `validation/structuring_validation_report.md` (precision 1.0, recall 0.8, n=11). The real ledger tier (10 accounts) is explicitly single-account only and does **not** feed this — see "What We Don't Claim" below. |
| **Drift anomalies** | The data-generating process itself changes over time, so a model trained on old data degrades | **Not covered.** No live drift monitoring exists today. A research script (`eval/verify_drift_signal.py`) explores this but currently runs on a synthetic fallback dataset with no engineered drift — see `validation/claims_ledger.md` for the full finding. This is a real gap, not a hidden one. |

## 2. What We Don't Claim (Slide / Talking Points)
- **Not an interconnected laundering network in the real ledger:** Disclosed as 10 largely independent accounts. The synthetic network (where collective-anomaly detection actually runs) is a separate, explicitly synthetic tier.
- **Not a reverse-engineering of anonymized PCA features into human meaning:** Respect the mathematical boundary of V1-V28 vs interpretable features (Time/Amount).
- **Not the first agentic fraud/AML tool:** Our competitive edge is execution rigor, strict trace-grounding, and verifiable honesty, not ungrounded marketing claims.
- **Not a production-audited compliance system:** A high-precision decision-support prototype.
- **Not drift-aware in production today:** Point, contextual, and collective anomalies are covered (with disclosed limits); drift detection is an identified, explicitly uncovered category, not a silent gap discovered later.

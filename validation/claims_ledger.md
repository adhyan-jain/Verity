# Claims Ledger — Independent Audit

Every quantitative or capability claim I could find in the codebase and pitch
materials, checked against something that actually runs. Status legend:
**BACKED** (verified against a live run/artifact), **STALE** (was true at
some point, superseded by later work), **UNBACKED** (no verifiable evidence
found), **UNRESOLVED** (needs a team decision, not something I can resolve
alone).

## 🚩 Flagged for team lead — action needed

| # | Issue | Why it matters |
|---|---|---|
| 1 | **`engines/fraud/model.pkl` is gitignored (`*.pkl` in `.gitignore`) and regenerated locally by whoever runs `train.py` on their own machine — there is no single canonical model artifact.** This machine's `model.pkl` right now is `v1.2-...` (LightGBM, SMOTE, precision 90.5%/recall 76.0%/F1 0.826/PR-AUC 0.805). `docs/FRAUD_MODEL_REPORT.md` documents a *different, chronologically later* `v2.0-...` (XGBoost, 250 rounds, different metrics) that was evidently trained and documented on a different machine/session — that artifact does not exist here. `docs/DATASETS.md` §2.2 documents a third, LightGBM run with yet another threshold. **All three could be independently true statements about three different points in time on three different machines** — none of them is provably "wrong," but there is no guarantee the model actually running during a live demo matches any specific number quoted in a doc. | This is the single most important finding in this ledger. Before presenting any specific metric number live, whoever runs the demo needs to either (a) run `python -m engines.fraud.train` fresh immediately beforehand and quote whatever `/health` reports at that moment, discarding all doc-quoted numbers, or (b) check a canonical model artifact into a location the whole team shares (contradicts the current `.gitignore` policy — a real trade-off to decide, not something I can decide unilaterally), or (c) stand up a shared training/CI step that produces one artifact everyone points to. I have not resolved this — it's a process/infrastructure decision above my role's scope. |
| 2 | `bank.xlsx` is described as "real" data in README.md, ARCHITECTURE.md, docs/DATASETS.md, docs/API_SPEC.md, PITCH_BRIEF.md — with **no checksum, source citation, or anonymization method documented anywhere**. | If a judge asks "how do we know this is real," there is currently no artifact to point to. Either produce one (source institution name even if redacted, a hash recorded at intake, or an explicit acknowledgment that provenance is asserted, not verified) or soften the claim. |
| 3 | No `v2.1` doc exists anywhere in the repo. | My task brief referenced auditing "the v2.1 doc" — it doesn't exist. `FRAUD_MODEL_REPORT.md` self-labels "Document Version 2.0.0" — possibly the intended reference, possibly a separate gap. Worth reconciling with whoever assigned that version number. |
| 4 | Test suite is currently **84 passed, 2 failed** (not 56/56 as an earlier doc snapshot may say — see entry below for why the count moved). The 2 failures are `test_get_shap_explanation_flagged`/`_clear` expecting a `risk_interval` field that's `None` because `engines/fraud/conformal.pkl` hasn't been generated in this environment. | Not a logic bug — `engines/fraud/conformal.py` logs exactly this at runtime: *"No conformal calibration artifact... run `python -m engines.fraud.conformal` after training the model."* Same root cause as item #1 (gitignored, per-machine model artifacts) — this is Person A's model-artifact domain, not mine to run. Flagging so it's fixed before a demo, not discovered during one. |
| 5 | `eval/verify_drift_signal.py` exists and runs, but **silently falls back to a fully synthetic, stationary Sparkov-format dataset** (`engines/fraud/datasets.py::generate_synthetic_kartik2112`) because the real Kaggle `fraudTrain.csv`/`fraudTest.csv` files aren't present in `data/raw/`. See detailed finding below — the current output does not demonstrate real drift-handling capability, and there is no console warning that the fallback occurred. | If this script's output is ever used to support a "we detect concept drift" pitch claim, that claim would currently be unbacked. Two independent fixes needed: (a) get the real dataset or make the fallback loud/documented, (b) even with real data, the current KS-test approach shows signs of producing false positives from multiple-testing without correction (see below) — that needs its own review before being pitch material. |
| 6 | `VAL-STR-TP-02` (see `validation/structuring_validation_report.md`): the structuring detector does not catch a single-originator repeated-below-threshold-transfer pattern (only multi-party fan-in). Unclear if this is an intentional scope decision or a gap. | Needs a decision from whoever owns `engines/typology/fatf_rules.py` on whether to close this or document it as an explicit scope boundary. |

## Fraud model metrics

**See flagged item #1 above — this is not a simple "pick the right number" situation.** Three different, each-internally-consistent metric sets exist because the model artifact is per-machine and gitignored, not because any doc is factually wrong about what it measured.

| Claim | Evidence | Dataset / split | Real/Synthetic | Status |
|---|---|---|---|---|
| LightGBM, precision 90.5%, recall 76.0%, F1 0.826, PR-AUC 0.805, ROC-AUC 0.984 | Loaded `engines/fraud/model.pkl` directly and read its stored `metrics`/`model_version` (`v1.2-20260918T070328Z`) — this is what `/api/v1/fraud/health` returns **on this machine, right now** | `data/raw/creditcard.csv`, chronological 80/20 split (test: 56,962 rows / 75 fraud) | Real (Kaggle creditcard.csv) | **BACKED for this environment only** — confirmed by loading the live artifact. Do not assume this matches what a different machine's `/health` would return. |
| XGBoost, precision 98.25%, recall 74.67%, F1 0.8485, PR-AUC 0.809 (`model_version v2.0-...`, chronologically the newest of the three) | `docs/FRAUD_MODEL_REPORT.md` | Same source dataset, XGBoost 250-round run, evidently trained on a different machine/session than this one | Real | **UNRESOLVED — cannot confirm reproducible here.** Not "stale" in the sense of being wrong; the artifact it describes simply isn't present in this environment to re-verify. |
| LightGBM (different threshold): precision 87.10%, recall 82.65%, F1 0.8482, PR-AUC 0.8735 | `docs/DATASETS.md` §2.2 | Same dataset, different training run/threshold calibration | Real | **UNRESOLVED** — same issue as above. |

**I have not edited either doc's numbers or added "superseded" banners** — on reflection, that would have been the wrong call given what I found: none of the three is demonstrably incorrect, they're likely three genuine runs from three points in time, and picking a "winner" is exactly the kind of production/model decision outside this role's scope. Flagged item #1 is the actual, more important issue underneath this.

## Test suite / CI claims

| Claim | Evidence | Status |
|---|---|---|
| "56/56 tests passing" (an earlier session's number, may still appear in some doc/commit message) | `PYTHONPATH=. uv run pytest -q` run just now: **84 passed, 2 failed** | **STALE** — the suite has grown substantially since that count (conformal prediction, dataset loaders, agent consistency eval all added tests). The 2 current failures are the conformal-artifact issue in flagged item #4 above, not a regression in older tests. |

## Typology / adversarial validation claims

| Claim | Evidence | Status |
|---|---|---|
| Typology detector accuracy on the (Person B–owned) adversarial set | Not independently re-verified in this pass — `data/synthetic/adversarial_set.json` is Person B's own artifact; re-running their own evaluator against their own set doesn't add independent evidence beyond what they've already reported. My independent contribution is the separate set in `validation/`, not re-verifying theirs. | Not re-checked — see `validation/structuring_validation_report.md` for the independent number instead: precision 1.0, recall 0.8 on structuring specifically. |
| Structuring detector, independently validated | `validation/structuring_validation_report.md` / `validation/structuring_validation_results.json` | **BACKED** — precision 1.0, recall 0.8, F1 0.889, n=11, with one disclosed gap and one disclosed independence caveat. |

## Drift detection

| Claim | Evidence | Dataset | Real/Synthetic | Status |
|---|---|---|---|---|
| (No explicit pitch claim currently exists — checked PITCH_BRIEF.md, ARCHITECTURE.md, docs/PITCH.md, README.md for "drift" language: none found beyond an unrelated lockfile-drift mention.) | `eval/verify_drift_signal.py` exists and runs; ran it live just now (24 monthly windows, 2019-2020) | **Silently falls back to `generate_synthetic_kartik2112()`** — the real Kaggle `fraudTrain.csv`/`fraudTest.csv` are not present in `data/raw/`, and the fallback prints no warning | Synthetic (undisclosed at runtime) | **UNBACKED as a capability claim** — see full finding below. Good news: no pitch doc currently overclaims this, so there's nothing to walk back, but it should stay unclaimed until fixed. |

**Full finding, since this was one of the specific risks I was asked to
check:** I ran `eval/verify_drift_signal.py` directly. It produced a table of
24 monthly KS-test results, alternating "YES (p<0.01)"/"NO" for roughly half
the months in a way that does not track any obvious trend (mean prediction
score bounces between 0.0031 and 0.0081 with no monotonic drift). I then
read `generate_synthetic_kartik2112()` (the fallback generator, not the
detector) and confirmed it has **no time-varying fraud rate or behavioral
parameters** — it's a stationary random process across the full simulated
2019-2020 window. Running a KS-test at α=0.01 across 24 independent monthly
comparisons without a multiple-testing correction is expected to produce
several false "drift detected" positives on genuinely stationary data by
chance alone, which is consistent with what was observed. **This script
currently demonstrates that the KS-test mechanically runs, not that the
system has evidenced drift-handling capability.** To make this claim-ready:
(1) source the real Sparkov/kartik2112 dataset so the drift, if any, is real
rather than a byproduct of a stationary generator; (2) make the synthetic
fallback loud (print a warning) rather than silent; (3) apply a multiple-
testing correction (e.g. Bonferroni) or a higher-confidence single
comparison rather than 24 independent monthly tests at a single α.

## GAN augmentation

| Claim | Evidence | Status |
|---|---|---|
| "Why not use GAN augmentation?" | No GAN/Proteus reference existed anywhere in this repo before this session (confirmed via repo-wide grep). See `demo/hostile_qa_rehearsal.md` for the answer, now backed by a real external citation (the user-provided Proteus paper, read in full) rather than an invented one. | **BACKED** (as of this session — previously **UNBACKED**, since the question was answerable only by inventing a citation) |

## V1-V28 feature disclosure

| Claim | Evidence | Status |
|---|---|---|
| "V1-V28 are PCA-transformed, non-interpretable, anonymized features" | Already correctly and repeatedly stated in `docs/DATASETS.md`, `docs/FRAUD_MODEL_REPORT.md`, `docs/PRD.md`, `docs/PITCH.md` | **BACKED** — this is genuinely well-handled already, nothing to fix. |

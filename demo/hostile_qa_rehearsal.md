# Hostile Question Rehearsal

Honest, rehearsed answers. Where the honest answer is a limitation, it's
stated as one — that's the point of rehearsing these in advance rather than
improvising under pressure.

## "What do V1-V28 actually represent?"

They're PCA-transformed features from the original Kaggle `creditcard.csv`
release — the dataset's own publisher ran principal component analysis on
the raw transaction/behavioral fields before releasing it, specifically to
protect the underlying cardholders' and merchants' privacy. We don't know
what V4 or V14 "mean" in human terms, and we don't pretend to. This is
already correctly documented in `docs/DATASETS.md`, `docs/FRAUD_MODEL_REPORT.md`,
and `docs/PRD.md` — the dashboard labels them "anonymized behavioral
signals" and never assigns them a human-readable meaning, unlike `Amount`
and `Time`, which are genuinely interpretable and labeled as such. If SHAP
says V14 is the top driver of a risk score, the honest answer to "why V14
specifically" is: we know it mattered mathematically, we don't know what it
represents physically, and we say exactly that rather than inventing a
plausible-sounding story.

## "Isn't your synthetic validation circular?"

For the synthetic typology network: yes, in the sense that the network
itself and the adversarial test set are both authored by the same
methodology, which is a real limitation, not a solved problem. What we've
done to push against that: the structuring detector specifically has a
**second, independently authored adversarial set**
(`validation/structuring_adversarial_set.json`), built by someone who
didn't write or read the detector's implementation, following FATF's public
definitions rather than the detector's own thresholds. It found a real
recall gap (a single-originator structuring variant isn't caught) and
reported it rather than hiding it. That's a genuine, if partial and
small-sample (n=11), attempt at breaking circularity — not a claim that
circularity is fully solved. For the fraud model: no circularity concern
there, since `creditcard.csv` is real transaction data with real fraud
labels from a public, external release.

## "This is just anomaly detection, what's novel?"

The honest framing: most fraud/AML pitches show one classifier and call it
done. Anomaly detection actually spans four distinct categories — point,
contextual, collective, and drift (see the taxonomy in `docs/PITCH.md`) —
and a single classifier only ever covers one of them well. What's actually
different here is naming that taxonomy explicitly and being honest about
coverage: point anomalies (single flagged transactions) and contextual
anomalies (per-account deviation from an account's own baseline) are
genuinely covered; collective anomalies (multi-entity laundering patterns)
are covered on a synthetic network with the circularity caveat above; drift
anomalies are **not** covered at all today (see the drift question below).
That's the actual honest pitch: a taxonomy with named, evidenced boundaries,
not a single model with an inflated claim.

## "Who actually buys this — big banks have separate fraud and AML teams?"

Correct, and that's exactly who this isn't built for. The target is
mid-size digital banks and NBFCs (non-bank lenders) that are too small to
staff two separate teams and two separate toolchains — one analyst
currently has to swivel-chair between a fraud dashboard and a separate
AML/ledger review tool. This is a real, named persona
(`PITCH_BRIEF.md`'s "Chitrita") and a real staffing-economics argument, not
a claim that this has been validated with an actual paying customer or
design partner — it hasn't, and we don't say it has.

## "Why no GAN augmentation?"

We looked at this and made a deliberate choice not to spend hours
rediscovering something already answered. There's a real, published,
full-scale ablation (Proteus, evaluating GAN-based minority-class
augmentation for imbalanced classification on 1.47M real network-traffic
rows) that found static, one-shot GAN augmentation **did not improve
macro-F1 over a plain baseline** (0.8915 vs. 0.8922 — the GAN-augmented run
was marginally *worse*), an explicitly reported honest negative result at
full data scale. Given that prior evidence, we used SMOTE instead — a much
simpler, well-understood oversampling method — rather than spending the
time to independently re-derive a result someone else already published.
That's a legitimate research decision (read the literature before repeating
an experiment), not an admission we didn't consider it.

## "Your ledger has 10 accounts. That's not a laundering network."

Correct, and we don't call it one. The 10-account real ledger tier is
explicitly single-account reconciliation only — balance breaks, timing
spikes, reversal outliers, each measured against that one account's own
history. The actual multi-hop, multi-entity laundering-network detection
(structuring, round-tripping, rapid layering) runs on a **separate,
explicitly synthetic** network (`data/synthetic/synthetic_network.json`),
not the 10 real accounts. These are two different tiers with two different
data-reality labels shown in the UI, and the pitch doesn't blur them
together — if a judge is confusing "10 real accounts" with "the laundering
network," that's worth correcting on the spot, not defending as if they
were the same claim.

## "What happens when the fraud pattern changes after deployment?"

The honest current answer: **there is no live drift monitoring today.**
There's a research/validation script (`eval/verify_drift_signal.py`) that
runs a KS-test drift check, but it currently falls back to a synthetic,
stationary dataset when the real multi-year dataset isn't present locally,
and even setting that aside, its current multiple-testing approach would
need a correction before its output could support a real claim (full
finding in `validation/claims_ledger.md`). What *does* exist: a versioned
retraining path — `engines/fraud/train.py` produces a timestamped,
versioned model artifact with recorded provenance (git SHA, dataset hash,
package versions) each time it's run, so retraining on new labeled data and
redeploying is mechanically straightforward. What doesn't exist yet is the
automatic trigger to know *when* to retrain. That's the accurate state of
the system, not a roadmap dressed up as a current feature.

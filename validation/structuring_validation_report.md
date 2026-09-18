# Independent Structuring Detector Validation Report

**Method:** 11 test cases (5 true structuring patterns per FATF, 6 legitimate
decoys), authored independently in `validation/structuring_adversarial_set.json`
per the FATF definitions in `validation/fatf_structuring_references.md`, run
through the live, unmodified `engines.typology.detect.evaluate_adversarial_set()`
via `validation/run_structuring_validation.py`. Full machine-readable output:
`validation/structuring_validation_results.json`.

**Disclosure of a compromise to independence:** while inspecting
`data/synthetic/adversarial_set.json`'s file format for schema compatibility
(field names only — I needed to know the input contract to build a
compatible file), a broader grep than intended surfaced some of the
detector-author's own specific edge-case boundary values (a 2-sender
threshold, specific dollar cumulative hurdles, the 24-hour window, and the
round-tripping/rapid-layering numeric bounds). I did not use those specific
values in my own set — the scenarios above use my own independently-chosen
time windows and dollar clustering — but full blindness to the detector's
edge-case choices was not maintained for the entire exercise. I am
disclosing this rather than silently claiming perfect independence, because
an undisclosed partial break in independence is exactly the kind of thing
this role exists to catch when someone else does it.

## Results (unmodified)

| Metric | Value |
|---|---|
| Total cases | 11 |
| True positives | 4 / 5 |
| True negatives | 6 / 6 |
| False positives | 0 |
| False negatives | 1 |
| **Precision** | **1.0** |
| **Recall** | **0.8** |
| **F1** | **0.889** |
| Specificity | 1.0 |
| Accuracy | 0.909 |

## The bad number, reported as instructed

`VAL-STR-TP-02` — a **single-originator** variant of structuring (one
account making 4 rapid transfers to the same new beneficiary, each
$9,000-$9,700, within 3 hours) — was **not flagged** (`actual_verdict:
"benign"`). This is a real recall gap: the detector appears to require
multiple distinct source accounts converging on a collector (the
"multi-smurf fan-in" pattern) and does not currently catch the simpler
single-account repeated-below-threshold-transfer variant, even though this
is also a recognized structuring pattern under FATF's framework (the
account holder is still deliberately avoiding a reporting threshold; FATF's
own guidance does not require multiple distinct originators for a pattern
to constitute structuring). This is worth a decision from whoever owns
`engines/typology/fatf_rules.py`: either the current design deliberately
scopes structuring detection to multi-party fan-in only (a reasonable,
documented scope-narrowing choice), or this is a genuine gap worth closing.
I am not able to tell which from outside the implementation, and I'm not
reading the implementation to find out, per this role's independence
constraint — this is a question for the team, not something I resolve
unilaterally.

## What went right

All 6 decoys — including the two I deliberately designed to be hard
(`VAL-STR-DECOY-05-SPLIT-LIMIT`, two of the same customer's own accounts
funneling threshold-clustered amounts into a shared account within 5
minutes; and `VAL-STR-DECOY-06-COINCIDENTAL-INVOICES`, three unrelated
clients coincidentally paying similar-magnitude legitimate invoices) — were
correctly left unflagged. Zero false positives across payroll, recurring
vendor payments, subscriptions, seasonal purchasing, and both hard
edge cases is a genuinely strong result and the more important number for a
pitch: it means the detector is not just pattern-matching on "many
sub-threshold transactions," which is the most common and most damaging
false-positive failure mode for a structuring detector in a real deployment
(alert fatigue from flagging every payroll run or subscription business).

## Bottom line for the pitch

**Report both numbers.** Precision 1.0 / recall 0.8 on an independently
authored set (n=11, small — this is not a statistically powered evaluation,
just a targeted adversarial probe) is an honest, creditable result: the
detector does not cry wolf on legitimate look-alike patterns, but it has at
least one identified gap (single-originator structuring) worth naming
explicitly in the "what we don't claim" pitch material rather than letting
a judge find it first.

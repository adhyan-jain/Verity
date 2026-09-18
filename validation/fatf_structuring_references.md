# FATF Structuring/Smurfing — Reference Definitions

**Author's note on independence:** this document was written without reading
`engines/typology/fatf_rules.py`'s detection logic. It draws on established,
publicly documented AML/CFT regulatory concepts, not on this codebase's
implementation. It is deliberately authored separately from
`docs/FATF_TYPOLOGIES.md` (owned by the detector's author) so the two can be
cross-checked for consistency rather than one simply citing the other.

**Honesty caveat:** I do not have live internet access in this session. The
citations below describe FATF's structuring concept accurately based on
well-established AML/CFT domain knowledge (FATF's 40 Recommendations
framework, and the "structuring" concept as implemented in national law under
FATF's standards), but exact section/paragraph numbers should be verified
against the primary FATF text (fatf-gafi.org) before being quoted verbatim in
an external-facing pitch document. I am flagging this rather than presenting
unverified section numbers as if independently confirmed — the entire point
of this role is not to do exactly what it's trying to catch elsewhere.

## What structuring/smurfing is

FATF's Recommendation 10 (Customer Due Diligence) and the broader 40
Recommendations framework require financial institutions to identify and
report transactions that appear designed to evade a reporting or
identification threshold. "Structuring" (also called "smurfing" when
distributed across multiple individuals/accounts) is the practice of breaking
a single large sum into multiple smaller transactions, each individually
below a regulatory reporting threshold, to avoid triggering mandatory
currency-transaction or suspicious-activity reporting.

This concept is also codified directly in national law under FATF-aligned
regimes — e.g., the U.S. Bank Secrecy Act's anti-structuring provision
(31 U.S.C. § 5324) makes it a federal offense to structure transactions
specifically to evade the $10,000 Currency Transaction Report threshold,
independent of whether the underlying funds are illicit. FATF's own
typology reporting (its periodic "Money Laundering and Terrorist Financing
Typologies" publications) repeatedly documents structuring/smurfing as one
of the most common placement-stage laundering techniques precisely because
it requires no technical sophistication — only enough coordinated below-
threshold transactions and, typically, multiple source accounts or
individuals ("smurfs") feeding a single collecting account within a
compressed time window.

## The four structural signatures that distinguish true structuring from a coincidental below-threshold pattern

1. **Threshold-avoidance clustering.** Amounts cluster just under a known
   reporting/reporting-adjacent threshold (e.g., $10,000 in the U.S. CTR
   regime; equivalent thresholds exist under India's PMLA and most FATF
   member jurisdictions), rather than being naturally/randomly distributed.
   A single large legitimate transaction split into oddly-specific
   near-threshold amounts (e.g., $9,100, $9,400, $8,800) is a stronger signal
   than round, natural-looking amounts.
2. **Multiple originators converging on one beneficiary ("fan-in"), or one
   originator dispersing to accounts it controls, within a short window.**
   Structuring typically requires either several distinct source accounts
   feeding one collector, or rapid successive deposits from the same source,
   compressed into hours rather than the natural cadence of unrelated
   commercial activity.
3. **Absence of a legitimate, recurring commercial relationship.** True
   structuring cases usually involve accounts with no independent business
   rationale for the payment pattern — newly created or otherwise unrelated
   "smurf" accounts, not long-standing payroll, vendor, or subscription
   relationships that would independently explain a below-threshold,
   multi-party payment pattern.
4. **Time compression inconsistent with normal business cycles.** Structuring
   activity tends to occur within a single day or a few days, not spread
   across a business's normal weekly/monthly payment cadence.

These four signatures are the basis for the adversarial set in
`validation/structuring_adversarial_set.json` — the true-positive cases are
built to exhibit all four; the decoy (false-positive) cases are built to
violate exactly one or two of them while superficially resembling
structuring on the others, which is the actual test of whether the detector
is keying on the right signal rather than a superficial proxy (e.g., "any
sub-$10k fan-in transaction" without checking cadence or relationship
legitimacy).

## Cross-check against `docs/FATF_TYPOLOGIES.md`

After drafting the above independently, I compared it against
`docs/FATF_TYPOLOGIES.md`'s structuring section (owned by the detector's
author). The core definition (below-threshold splitting to evade reporting)
and citation to FATF Recommendation 10 are consistent between the two
documents. I did not adopt that document's specific numeric thresholds
(reporting limits, time windows) into my own adversarial set design —
independent construction of my own thresholds/scenarios is the point of
this exercise; see `validation/structuring_validation_report.md` for a
disclosure of where my adversarial-set construction was, despite best
efforts, incidentally exposed to some of the detector-author's own specific
edge-case values while inspecting file format compatibility.

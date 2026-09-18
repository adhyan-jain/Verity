"""
Independent validation runner for the structuring detector.

Calls the existing, unmodified typology evaluation function
(engines.typology.detect.evaluate_adversarial_set) against
validation/structuring_adversarial_set.json — an independently authored
adversarial set (see validation/fatf_structuring_references.md for the FATF
basis and validation/structuring_adversarial_set.json's test_rationale
fields for per-case reasoning).

This script writes no detection logic of its own: it only invokes the
detector's own evaluation entrypoint against a different input file, exactly
as it already supports via its `adversarial_file` parameter.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.typology.detect import evaluate_adversarial_set

VALIDATION_SET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "structuring_adversarial_set.json"
)
RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "structuring_validation_results.json"
)


def main() -> None:
    with open(VALIDATION_SET_PATH) as f:
        cases = json.load(f)

    results = evaluate_adversarial_set(adversarial_file=VALIDATION_SET_PATH)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Ran {len(cases)} independently-authored cases through the live detector.")
    print(json.dumps(results, indent=2))
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

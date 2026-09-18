"""
FATF Typology Detector Service & Evaluator.
Person B: Evaluates synthetic and graph networks against FATF rules,
outputting validated TypologyFlag records.
"""

import json
import os
from typing import Any

try:
    from .fatf_rules import (
        build_networkx_graph,
        detect_rapid_layering,
        detect_round_tripping,
        detect_structuring,
    )
except (ImportError, ValueError):
    from fatf_rules import (
        build_networkx_graph,
        detect_rapid_layering,
        detect_round_tripping,
        detect_structuring,
    )

DEFAULT_NETWORK_FILE = "data/synthetic/synthetic_network.json"


def load_synthetic_network(file_path: str = DEFAULT_NETWORK_FILE) -> dict[str, Any]:
    """
    Loads synthetic network dataset from JSON.
    """
    if not os.path.exists(file_path):
        from data.synthetic.generate_network import generate_fatf_network

        return generate_fatf_network(output_file=file_path)

    with open(file_path, "r") as f:
        return json.load(f)


def detect_all_typologies(
    network_data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Runs all FATF typology detection algorithms on the given network.
    Returns list of TypologyFlag compliant records.
    """
    if network_data is None:
        network_data = load_synthetic_network()

    edges = network_data.get("edges", [])
    if not edges:
        return []

    G = build_networkx_graph(edges)

    structuring_flags = detect_structuring(G)
    round_trip_flags = detect_round_tripping(G)
    layering_flags = detect_rapid_layering(G)

    all_flags = structuring_flags + round_trip_flags + layering_flags
    all_flags.sort(key=lambda x: x["confidence"], reverse=True)
    return all_flags


def evaluate_adversarial_set(
    adversarial_file: str = "data/synthetic/adversarial_set.json",
) -> dict[str, Any]:
    """
    Evaluates detector accuracy, precision, recall, specificity, and false-positive rates on the held-out adversarial set.
    """
    if not os.path.exists(adversarial_file):
        return {"error": "Adversarial test set not found"}

    with open(adversarial_file, "r") as f:
        test_cases = json.load(f)

    results = []
    tp = fp = tn = fn = 0

    for case in test_cases:
        test_id = case["test_id"]
        expected = case["expected_verdict"]
        edges = case.get("edges", [])

        G = build_networkx_graph(edges)
        detected = (
            detect_structuring(G) + detect_round_tripping(G) + detect_rapid_layering(G)
        )

        actual_verdict = "flagged" if len(detected) > 0 else "benign"
        is_correct = actual_verdict == expected

        if expected == "flagged":
            if actual_verdict == "flagged":
                tp += 1
            else:
                fn += 1
        else:
            if actual_verdict == "benign":
                tn += 1
            else:
                fp += 1

        results.append(
            {
                "test_id": test_id,
                "description": case["description"],
                "expected_verdict": expected,
                "actual_verdict": actual_verdict,
                "passed": is_correct,
                "detected_flags": [d["typology"] for d in detected],
            }
        )

    total = len(test_cases)
    accuracy = (tp + tn) / max(1, total)
    precision = tp / max(1, tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / max(1, tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / max(1, tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / max(1e-10, precision + recall)

    return {
        "total_test_cases": total,
        "passed_cases": tp + tn,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
        "accuracy": round(accuracy, 4),
        "details": results,
    }



if __name__ == "__main__":
    print("Running FATF Typology Detection on Synthetic Network...")
    flags = detect_all_typologies()
    print(f"Found {len(flags)} FATF typology flags:")
    for f in flags:
        print(
            f"  [{f['typology'].upper()}] {f['flag_id']} (Conf: {f['confidence']}) -> Accounts: {f['involved_accounts']}, Evt Txns: {f['evidence_transaction_ids']}"
        )

    print("\nEvaluating Adversarial Hold-Out Test Set (50 cases)...")
    eval_res = evaluate_adversarial_set()
    print(
        f"Results: Precision: {eval_res['precision'] * 100:.1f}%, Recall: {eval_res['recall'] * 100:.1f}%, "
        f"F1: {eval_res['f1']:.4f}, Specificity: {eval_res['specificity'] * 100:.1f}%, Accuracy: {eval_res['accuracy'] * 100:.1f}% "
        f"(TP: {eval_res['true_positives']}, TN: {eval_res['true_negatives']}, FP: {eval_res['false_positives']}, FN: {eval_res['false_negatives']})"
    )
    for d in eval_res["details"]:
        status_icon = "PASS" if d["passed"] else "FAIL"
        print(
            f"  [{status_icon}] {d['test_id']}: {d['description']} -> Expected: {d['expected_verdict']}, Got: {d['actual_verdict']}"
        )


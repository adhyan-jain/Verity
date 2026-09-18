#!/usr/bin/env python
"""
End-to-end smoke test for Verity. Starts all four backend services against
the real datasets (no mocks), exercises the critical analyst flows over real
HTTP, then tears everything down. Exit code 0 = all checks passed.

Usage:
    python scripts/smoke_test.py
"""

import os
import sys
import traceback

import requests
from _services import (
    ensure_fraud_conformal_calibrated,
    load_env_file,
    start_all_backend_services,
    stop_all,
)

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label} {detail}")
        FAILURES.append(f"{label} {detail}".strip())


def run_checks() -> None:
    fraud_key = os.environ.get("FRAUD_API_KEY", "")
    fraud_headers = {"X-API-Key": fraud_key}

    print("\n1. Fraud engine — real card transaction + SHAP explanation")
    resp = requests.get(
        "http://127.0.0.1:8001/api/v1/fraud/transaction/TX-CARD-9842",
        headers=fraud_headers,
        timeout=10,
    )
    check(
        "get_transaction returns 200",
        resp.status_code == 200,
        f"(got {resp.status_code})",
    )
    body = resp.json()
    check("transaction id echoes back", body.get("id") == "TX-CARD-9842")
    check("tier is real_card", body.get("tier") == "real_card")

    resp = requests.get(
        "http://127.0.0.1:8001/api/v1/fraud/explain/TX-CARD-9842",
        headers=fraud_headers,
        timeout=15,
    )
    check(
        "get_shap_explanation returns 200",
        resp.status_code == 200,
        f"(got {resp.status_code})",
    )
    explanation = resp.json()
    check(
        "risk_score present and in [0,1]",
        0.0 <= explanation.get("risk_score", -1) <= 1.0,
    )
    check("top_factors non-empty", len(explanation.get("top_factors", [])) > 0)

    print("\n1b. Fraud engine — rejects requests without a valid API key")
    resp = requests.get(
        "http://127.0.0.1:8001/api/v1/fraud/transaction/TX-CARD-9842", timeout=10
    )
    check(
        "unauthenticated request is rejected (401)",
        resp.status_code == 401,
        f"(got {resp.status_code})",
    )

    print("\n2. Ledger engine — real bank.xlsx accounts, timeline, walk")
    resp = requests.get("http://127.0.0.1:8002/api/v1/ledger/accounts", timeout=45)
    check(
        "list_accounts returns 200",
        resp.status_code == 200,
        f"(got {resp.status_code})",
    )
    accounts = resp.json()
    check("at least one real account parsed", len(accounts) > 0)
    account_id = accounts[0]["account_id"] if accounts else None

    if account_id:
        resp = requests.get(
            f"http://127.0.0.1:8002/api/v1/ledger/timeline/{account_id}", timeout=45
        )
        check(
            "get_timeline returns 200 for a real account",
            resp.status_code == 200,
            f"(got {resp.status_code})",
        )
        timeline = resp.json()
        check("timeline has transactions", len(timeline.get("transactions", [])) > 0)

        resp = requests.get(
            f"http://127.0.0.1:8002/api/v1/ledger/walk/{account_id}", timeout=10
        )
        check(
            "walk_ledger_graph returns 200",
            resp.status_code == 200,
            f"(got {resp.status_code})",
        )

    print("\n3. Typology engine — synthetic FATF network + flags")
    resp = requests.get("http://127.0.0.1:8003/api/v1/typology/network", timeout=10)
    check(
        "get_synthetic_network returns 200",
        resp.status_code == 200,
        f"(got {resp.status_code})",
    )
    network = resp.json()
    check(
        "network has nodes and edges",
        len(network.get("nodes", [])) > 0 and len(network.get("edges", [])) > 0,
    )

    resp = requests.get("http://127.0.0.1:8003/api/v1/typology/flags", timeout=10)
    check(
        "list_typology_flags returns 200",
        resp.status_code == 200,
        f"(got {resp.status_code})",
    )
    check("at least one FATF typology flagged", len(resp.json()) > 0)

    print(
        "\n4. Agent core — full investigation loop (live mode, real fraud+ledger+typology calls)"
    )
    resp = requests.post(
        "http://127.0.0.1:8000/api/v1/agent/investigate",
        json={
            "case_id": "SMOKE-CARD-001",
            "transaction_id": "TX-CARD-9842",
            "tier_origin": "real_card",
        },
        timeout=30,
    )
    check(
        "investigate returns 200", resp.status_code == 200, f"(got {resp.status_code})"
    )
    case = resp.json()
    check("case has at least one trace event", len(case.get("trace_events", [])) > 0)
    check("narrative is non-empty", len(case.get("narrative", "")) > 0)
    check("risk_score present", "risk_score" in case)

    # Regression coverage: get_transaction must route to the engine that owns
    # the ID (fraud/ledger/typology), not always the fraud engine — see
    # ARCHITECTURE.md for the bug this caught during integration.
    resp = requests.post(
        "http://127.0.0.1:8000/api/v1/agent/investigate",
        json={
            "case_id": "SMOKE-SYNTH-001",
            "transaction_id": "TX-SYNTH-5501",
            "tier_origin": "synthetic_network",
        },
        timeout=30,
    )
    synth_case = resp.json()
    check(
        "synthetic-tier investigation stays on synthetic_network (not mis-routed to fraud engine)",
        synth_case.get("tier_origin") == "synthetic_network",
        f"(got {synth_case.get('tier_origin')})",
    )
    check(
        "synthetic-tier investigation calls walk_graph",
        any(
            e.get("tool_called") == "walk_graph"
            for e in synth_case.get("trace_events", [])
        ),
    )

    resp = requests.post(
        "http://127.0.0.1:8000/api/v1/agent/investigate",
        json={
            "case_id": "SMOKE-LEDGER-001",
            "transaction_id": "TX-LEDGER-3011",
            "tier_origin": "real_ledger",
        },
        timeout=30,
    )
    ledger_case = resp.json()
    check(
        "ledger-tier investigation stays on real_ledger (not mis-routed to fraud engine)",
        ledger_case.get("tier_origin") == "real_ledger",
        f"(got {ledger_case.get('tier_origin')})",
    )

    print("\n5. Agent core — counterfactual (model-backed re-scoring)")
    # TX-CARD-9842 is the demo narrative's flagship card case, but "9842" is
    # also parsed as a literal row index (engines/fraud/api.py::parse_row_index),
    # and that real row happens to be a low-risk, non-fraud transaction — its
    # score floors out near zero with nothing left to reduce. TX-CARD-623 is a
    # real Class=1 (fraud) row, so an amount cut has a measurable effect to
    # assert against. See ARCHITECTURE.md "Remaining gaps" for the ID overload.
    resp = requests.post(
        "http://127.0.0.1:8000/api/v1/agent/counterfactual",
        json={"transaction_id": "TX-CARD-623", "parameter_overrides": {"Amount": 5.0}},
        timeout=15,
    )
    check(
        "counterfactual returns 200",
        resp.status_code == 200,
        f"(got {resp.status_code})",
    )
    cf = resp.json()
    check(
        "lowering the amount lowers the recalculated risk score",
        cf.get("recalculated_risk_score", 1.0) < cf.get("original_risk_score", 0.0),
        f"(original={cf.get('original_risk_score')}, recalculated={cf.get('recalculated_risk_score')})",
    )

    print("\n6. Agent core — chat, including cached fallback for a benchmark question")
    resp = requests.post(
        "http://127.0.0.1:8000/api/v1/agent/chat",
        json={"case_id": "SMOKE-CARD-001", "query": "why was this flagged"},
        timeout=25,
    )
    check("chat returns 200", resp.status_code == 200, f"(got {resp.status_code})")
    chat = resp.json()
    check(
        "cached benchmark answer is used and disclosed",
        chat.get("is_fallback") is True and bool(chat.get("fallback_notice")),
    )


def main() -> int:
    load_env_file()
    ensure_fraud_conformal_calibrated()

    print("Starting backend services for smoke test...")
    procs = start_all_backend_services()
    try:
        run_checks()
    except Exception:
        traceback.print_exc()
        FAILURES.append("unhandled exception during smoke test (see traceback above)")
    finally:
        stop_all(procs)

    print(f"\n{'=' * 60}")
    if FAILURES:
        print(f"SMOKE TEST FAILED — {len(FAILURES)} check(s) failed:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("SMOKE TEST PASSED — all critical paths verified end to end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

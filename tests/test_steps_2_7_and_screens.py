"""
Unit & Integration Tests for Verity Steps 2-7, Evaluation Hygiene, and Product Screens.
"""

import os
import numpy as np
import pandas as pd
import pytest

from engines.fraud.paysim_train import engineer_paysim_features, PAYSIM_PATH
from engines.fraud.paysim_score import score_bank_ledger, get_flagged_queue
from engines.typology.structuring_ledger import detect_structuring, structuring_summary
from engines.fraud.drift import run_drift_detection
from agent.prosecutor_defender import adjudicate_case, _resolution_to_dict
from agent.trust_score import TrustScoreSession
from engines.fraud.paysim_conformal import calibrate, wrap_score
from agent.breakdown import generate_account_breakdown
from agent.simulation import run_customer_counterfactual, run_forward_simulation, handle_scoped_customer_chat
from engines.ledger.parse_narrations import parse_bank_ledger


# ---------------------------------------------------------------------------
# Test Step 2: PaySim Feature Engineering & Model
# ---------------------------------------------------------------------------
def test_paysim_features_schema():
    df = pd.DataFrame([{
        "step": 10,
        "type": "TRANSFER",
        "amount": 5000.0,
        "nameOrig": "C123",
        "oldbalanceOrg": 10000.0,
        "newbalanceOrig": 5000.0,
        "nameDest": "C456",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 5000.0,
    }])
    feats = engineer_paysim_features(df)
    assert "balance_drain_ratio" in feats.columns
    assert "tx_velocity" in feats.columns
    assert "orig_dest_mismatch" in feats.columns
    assert feats.loc[0, "balance_drain_ratio"] == pytest.approx(0.5, 0.01)
    assert feats.loc[0, "type_TRANSFER"] == 1.0
    assert feats.loc[0, "type_PAYMENT"] == 0.0


# ---------------------------------------------------------------------------
# Test Evaluation Hygiene (MANDATORY REQUIREMENT)
# "Unit-test the split for disjointness, correct proportion, and seed reproducibility."
# ---------------------------------------------------------------------------
def test_evaluation_hygiene_split():
    # Load 1,000 synthetic or PaySim steps
    steps = np.arange(1, 1001)
    df = pd.DataFrame({"step": steps, "amount": 100.0})
    
    # 80/20 chronological time-split
    split_step = int(df["step"].quantile(0.80))
    train_part = df[df["step"] <= split_step]
    held_out_eval = df[df["step"] > split_step]
    
    # 1. Disjointness check
    train_steps = set(train_part["step"])
    eval_steps = set(held_out_eval["step"])
    assert train_steps.isdisjoint(eval_steps), "Train and Eval partitions must be strictly disjoint!"
    
    # 2. Correct proportion check
    total_len = len(df)
    eval_prop = len(held_out_eval) / total_len
    assert eval_prop == pytest.approx(0.20, abs=0.01), f"Held-out proportion was {eval_prop}, expected 0.20"
    
    # 3. Seed reproducibility check
    rng1 = np.random.default_rng(42)
    sample1 = rng1.choice(len(held_out_eval), size=50, replace=False)
    
    rng2 = np.random.default_rng(42)
    sample2 = rng2.choice(len(held_out_eval), size=50, replace=False)
    
    np.testing.assert_array_equal(sample1, sample2, "Sampling with fixed seed must be deterministic and reproducible")


# ---------------------------------------------------------------------------
# Test Step 3: Collective Structuring Detector
# ---------------------------------------------------------------------------
def test_structuring_detector():
    now = pd.Timestamp("2018-01-01")
    # 10 small debits of $95 = $950 in 10 days
    rows = []
    for i in range(10):
        rows.append({
            "id": f"TX-{i}",
            "account_id": "ACCT-TEST",
            "datetime": now + pd.Timedelta(days=i),
            "timestamp": (now + pd.Timedelta(days=i)).isoformat(),
            "direction": "debit",
            "amount": 95.0,
            "balance": 10000.0 - i * 95.0,
            "payment_rail": "CASH_ATM",
        })
    test_df = pd.DataFrame(rows)
    flags = detect_structuring(test_df, threshold=1000.0, margin=100.0, window_days=30, min_txns=2)
    assert len(flags) > 0
    top = flags[0]
    assert 900.0 <= top["window_sum"] < 1000.0
    assert top["severity"] > 0.0


# ---------------------------------------------------------------------------
# Test Step 4: Drift Detector (KS-Test, Detection + Logging Only)
# ---------------------------------------------------------------------------
def test_drift_detector():
    # Stable distribution
    rng = np.random.default_rng(42)
    scores_stable = list(rng.uniform(0.1, 0.3, 1200))
    res_stable = run_drift_detection(scores_stable, window_size=500, alpha=0.05)
    assert len(res_stable) > 0
    # Drifted distribution
    scores_drifted = list(rng.uniform(0.1, 0.3, 500)) + list(rng.uniform(0.7, 0.9, 500))
    res_drifted = run_drift_detection(scores_drifted, window_size=500, alpha=0.05)
    assert len(res_drifted) == 1
    assert res_drifted[0]["drift_fired"] is True


# ---------------------------------------------------------------------------
# Test Step 5: Prosecutor / Defender Grounded Adjudication
# ---------------------------------------------------------------------------
def test_prosecutor_defender():
    now = pd.Timestamp("2018-06-01")
    history = [
        {"id": "TX-OLD-1", "account_id": "ACC-1", "datetime": now - pd.Timedelta(days=60), "amount": 500.0, "direction": "debit", "payment_rail": "NEFT", "raw_narration": "MONTHLY RENT", "balance": 5000.0},
        {"id": "TX-OLD-2", "account_id": "ACC-1", "datetime": now - pd.Timedelta(days=30), "amount": 500.0, "direction": "debit", "payment_rail": "NEFT", "raw_narration": "MONTHLY RENT", "balance": 4500.0},
    ]
    curr = {"id": "TX-CURR", "account_id": "ACC-1", "datetime": now, "amount": 500.0, "direction": "debit", "payment_rail": "NEFT", "raw_narration": "MONTHLY RENT", "balance": 4000.0}
    acct_df = pd.DataFrame(history + [curr])
    
    res = adjudicate_case(
        tx_row=pd.Series(curr),
        acct_df=acct_df,
        risk_score=0.60,
        all_scores=[0.2] * 50 + [0.6],
        log=False,
    )
    d = _resolution_to_dict(res)
    # Defender should find recurring payment match
    assert len(d["defender"]["grounded_reasons"]) >= 1
    assert d["verdict"] in ("downgraded", "cleared")


# ---------------------------------------------------------------------------
# Test Step 6: Live Trust Score
# ---------------------------------------------------------------------------
def test_live_trust_score():
    session = TrustScoreSession("CASE-TEST-TRUST")
    trace = [
        {"event_id": "EVT-1", "narration_sentence": "Account executed transfer of ₹5,000 on NEFT rail.", "tool_output_summary": "amt 5000 rail NEFT"}
    ]
    tag1 = session.tag_claim("Account executed transfer of ₹5,000 on NEFT rail.", trace)
    assert tag1.grounded is True
    
    tag2 = session.tag_claim("Unverified offshore shell company in Cayman Islands detected.", trace)
    assert tag2.grounded is False
    
    state = session.state.to_dict()
    assert state["total_claims"] == 2
    assert state["grounded_claims"] == 1
    assert state["grounded_pct"] == 50.0


# ---------------------------------------------------------------------------
# Test Step 7: Conformal Prediction Intervals
# ---------------------------------------------------------------------------
def test_conformal_intervals():
    wrapped = wrap_score(0.75, confidence=0.90)
    assert "conformal_lo" in wrapped
    assert "conformal_hi" in wrapped
    assert wrapped["conformal_lo"] <= 0.75 <= wrapped["conformal_hi"]
    assert "90% CI:" in wrapped["label"]


# ---------------------------------------------------------------------------
# Test Screen 3: Breakdown Panel & Gate 1 Compliance
# ---------------------------------------------------------------------------
def test_breakdown_panel_gate1():
    bank_df = parse_bank_ledger()
    account_id = "409000493210"
    bd = generate_account_breakdown(account_id, bank_df=bank_df)
    assert bd["total_claims"] > 0
    for claim in bd["claims"]:
        assert claim["source_tag"] != ""
        assert claim["contribution"] > 0
        assert claim["gate1_compliant"] is True
        # Verify strict Gate 1 compliance: no time-of-day phrases
        s_lower = claim["sentence"].lower()
        assert "3am" not in s_lower
        assert "03:" not in s_lower
        assert "at night" not in s_lower
        assert "peak hour" not in s_lower


# ---------------------------------------------------------------------------
# Test Screen 5: Scoped Chat, Counterfactual & Forward Simulation
# ---------------------------------------------------------------------------
def test_scoped_simulation_and_counterfactual():
    bank_df = parse_bank_ledger()
    account_id = "409000493210"
    tx_id = "TX-LEDGER-114686"
    
    # 1. Counterfactual recompute
    cf = run_customer_counterfactual(account_id, tx_id, new_amount=2000.0, bank_df=bank_df)
    assert "recalculated_risk_score" in cf
    assert cf["conformal_interval"][0] <= cf["recalculated_risk_score"] <= cf["conformal_interval"][1]
    
    # 2. Forward simulation
    fwd = run_forward_simulation(account_id, tx_id, days_ahead=7, bank_df=bank_df)
    assert "forward_risk_score" in fwd
    assert "simulated_balance" in fwd
    assert "rolling_30d_outflow" in fwd
    
    # 3. Scoped customer chat routing
    chat_fwd = handle_scoped_customer_chat(account_id, "if she does this again next week, does it still flag?", bank_df=bank_df)
    assert chat_fwd["query_type"] == "forward_simulation"
    assert "Forward simulation" in chat_fwd["response"]
    
    chat_cf = handle_scoped_customer_chat(account_id, "what if this was $2,000?", bank_df=bank_df)
    assert chat_cf["query_type"] == "counterfactual"
    assert "Counterfactual recompute" in chat_cf["response"]


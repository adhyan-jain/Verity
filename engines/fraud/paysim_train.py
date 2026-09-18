"""
PaySim Fraud Model Training Pipeline — Step 2.
Trains LightGBM AND Random Forest on PaySim data using a chronological
time-split (step column). Reports macro-F1, weighted-F1, precision/recall
at a fixed alert budget for both models side-by-side.

Features engineered:
  1. balance_drain_ratio   — how much of orig's opening balance left the account
  2. tx_velocity           — step-normalised proxy for velocity (log1p step)
  3. orig_dest_mismatch    — dest closing balance fails to reflect inflow
  4. amount_log            — log-scaled amount (heavy-tailed distribution)
  5. type_* dummies        — transaction type one-hot

Run from project root:
    python -m engines.fraud.paysim_train
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pickle
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import lightgbm as lgb

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

PAYSIM_PATH = os.path.join(
    _ROOT, "data", "Pay_sim", "PS_20174392719_1491204439457_log.csv"
)
MODEL_OUTPUT_PATH = os.path.join(_HERE, "paysim_model.pkl")
METRICS_OUTPUT_PATH = os.path.join(_HERE, "paysim_benchmark.json")

# Fixed alert budget: top-K fraction of test set we treat as "alerts"
ALERT_BUDGET_FRACTION = 0.01   # top 1% by predicted probability
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_paysim_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces the canonical PaySim feature matrix.
    Required input columns:
        step, type, amount,
        oldbalanceOrg, newbalanceOrig,
        oldbalanceDest, newbalanceDest
    Returns a DataFrame of floats with no NaNs.
    """
    out = pd.DataFrame(index=df.index)

    # 1. Balance-drain ratio: fraction of orig opening balance that left
    #    Clamped to [0, 1] to handle edge-cases where opening balance = 0
    orig_open = df["oldbalanceOrg"].clip(lower=1e-2)
    out["balance_drain_ratio"] = (df["amount"] / orig_open).clip(0, 1)

    # 2. Transaction velocity proxy: log1p(step) — later steps = denser
    #    network history, correlated with fraud campaigns
    out["tx_velocity"] = np.log1p(df["step"])

    # 3. Orig-vs-dest balance mismatch:
    #    Expected: newbalanceDest ≈ oldbalanceDest + amount
    #    Fraud: dest doesn't gain what orig lost (cash-out laundering)
    expected_dest = df["oldbalanceDest"] + df["amount"]
    out["orig_dest_mismatch"] = (expected_dest - df["newbalanceDest"]).abs()
    # Log-scale (very heavy-tailed)
    out["orig_dest_mismatch"] = np.log1p(out["orig_dest_mismatch"])

    # 4. Log amount
    out["amount_log"] = np.log1p(df["amount"])

    # 5. Orig opening balance (log-scaled)
    out["orig_open_log"] = np.log1p(df["oldbalanceOrg"])

    # 6. Orig closing balance zero flag (common in fraud TRANSFER/CASH_OUT)
    out["orig_close_zero"] = (df["newbalanceOrig"] == 0).astype(float)

    # 7. Type dummies  — PAYMENT, TRANSFER, CASH_OUT, CASH_IN, DEBIT
    type_dummies = pd.get_dummies(df["type"], prefix="type", dtype=float)
    for col in ["type_PAYMENT", "type_TRANSFER", "type_CASH_OUT",
                "type_CASH_IN", "type_DEBIT"]:
        out[col] = type_dummies.get(col, 0.0)

    return out.fillna(0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _pkg_versions() -> dict[str, str]:
    vers: dict[str, str] = {}
    for pkg in ("lightgbm", "scikit-learn", "pandas", "numpy"):
        try:
            vers[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            vers[pkg] = "unknown"
    return vers


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Precision in the top-k scored samples."""
    top_k_idx = np.argsort(y_score)[::-1][:k]
    return float(y_true[top_k_idx].mean())


def _recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Recall for the top-k scored samples."""
    top_k_idx = np.argsort(y_score)[::-1][:k]
    tp = int(y_true[top_k_idx].sum())
    total_pos = int(y_true.sum())
    return float(tp / max(1, total_pos))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_paysim_models(
    data_path: str = PAYSIM_PATH,
    output_path: str = MODEL_OUTPUT_PATH,
    metrics_path: str = METRICS_OUTPUT_PATH,
    random_state: int = RANDOM_SEED,
    alert_budget_fraction: float = ALERT_BUDGET_FRACTION,
    chunksize: int = 500_000,
) -> dict[str, Any]:
    """
    Trains LightGBM and Random Forest on PaySim with a chronological time-split.
    Returns the artifact dict (also saved to output_path).
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"PaySim CSV not found at: {data_path}")

    print(f"[PaySim] Loading data from {data_path} ...")
    t_load = time.time()

    # Read in chunks to be memory-efficient on 6M+ row file
    chunks = []
    for chunk in pd.read_csv(data_path, chunksize=chunksize,
                             dtype={"isFraud": "int8", "isFlaggedFraud": "int8"}):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    print(f"[PaySim] Loaded {len(df):,} rows in {time.time()-t_load:.1f}s")
    print(f"[PaySim] Overall fraud rate: {df['isFraud'].mean():.4%}")

    # Chronological split on `step`
    split_step = int(df["step"].quantile(0.80))
    train_df = df[df["step"] <= split_step].copy()
    test_df  = df[df["step"] >  split_step].copy()

    print(f"[PaySim] Train: {len(train_df):,} rows (steps 1–{split_step}), "
          f"fraud={train_df['isFraud'].mean():.4%}")
    print(f"[PaySim] Test:  {len(test_df):,} rows (steps {split_step+1}+), "
          f"fraud={test_df['isFraud'].mean():.4%}")

    X_train = engineer_paysim_features(train_df)
    y_train = train_df["isFraud"].values.astype(int)
    X_test  = engineer_paysim_features(test_df)
    y_test  = test_df["isFraud"].values.astype(int)

    feature_names = X_train.columns.tolist()
    print(f"[PaySim] Features ({len(feature_names)}): {feature_names}")

    alert_k = max(1, int(len(y_test) * alert_budget_fraction))
    print(f"[PaySim] Alert budget: top {alert_k:,} ({alert_budget_fraction:.1%} of test)")

    results: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # A. LightGBM
    # ------------------------------------------------------------------
    print("\n[PaySim] Training LightGBM ...")
    scale_pos_weight = float((y_train == 0).sum()) / max(1, (y_train == 1).sum())
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=8,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    t0 = time.time()
    lgb_clf.fit(X_train, y_train)
    lgb_train_sec = round(time.time() - t0, 2)

    lgb_proba = lgb_clf.predict_proba(X_test)[:, 1]
    lgb_pred  = (lgb_proba >= 0.5).astype(int)

    results["lightgbm"] = {
        "macro_f1":    round(float(f1_score(y_test, lgb_pred, average="macro",    zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_test, lgb_pred, average="weighted", zero_division=0)), 4),
        "precision":   round(float(precision_score(y_test, lgb_pred, zero_division=0)), 4),
        "recall":      round(float(recall_score(y_test, lgb_pred, zero_division=0)), 4),
        "roc_auc":     round(float(roc_auc_score(y_test, lgb_proba)), 4),
        "precision_at_k": round(_precision_at_k(y_test, lgb_proba, alert_k), 4),
        "recall_at_k":    round(_recall_at_k(y_test, lgb_proba, alert_k), 4),
        "alert_k":     alert_k,
        "train_sec":   lgb_train_sec,
    }
    print(f"[LightGBM] macro-F1={results['lightgbm']['macro_f1']:.4f}  "
          f"weighted-F1={results['lightgbm']['weighted_f1']:.4f}  "
          f"P@K={results['lightgbm']['precision_at_k']:.4f}  "
          f"R@K={results['lightgbm']['recall_at_k']:.4f}")

    # ------------------------------------------------------------------
    # B. Random Forest
    # ------------------------------------------------------------------
    print("\n[PaySim] Training Random Forest (100 trees, seed=42) ...")
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    t0 = time.time()
    rf_clf.fit(X_train, y_train)
    rf_train_sec = round(time.time() - t0, 2)

    rf_proba = rf_clf.predict_proba(X_test)[:, 1]
    rf_pred  = (rf_proba >= 0.5).astype(int)

    results["random_forest"] = {
        "macro_f1":    round(float(f1_score(y_test, rf_pred, average="macro",    zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_test, rf_pred, average="weighted", zero_division=0)), 4),
        "precision":   round(float(precision_score(y_test, rf_pred, zero_division=0)), 4),
        "recall":      round(float(recall_score(y_test, rf_pred, zero_division=0)), 4),
        "roc_auc":     round(float(roc_auc_score(y_test, rf_proba)), 4),
        "precision_at_k": round(_precision_at_k(y_test, rf_proba, alert_k), 4),
        "recall_at_k":    round(_recall_at_k(y_test, rf_proba, alert_k), 4),
        "alert_k":     alert_k,
        "train_sec":   rf_train_sec,
    }
    print(f"[RF]       macro-F1={results['random_forest']['macro_f1']:.4f}  "
          f"weighted-F1={results['random_forest']['weighted_f1']:.4f}  "
          f"P@K={results['random_forest']['precision_at_k']:.4f}  "
          f"R@K={results['random_forest']['recall_at_k']:.4f}")

    # ------------------------------------------------------------------
    # Side-by-side summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"{'Metric':<25} {'LightGBM':>12} {'Random Forest':>14}")
    print("-" * 60)
    for key in ["macro_f1", "weighted_f1", "precision", "recall",
                "roc_auc", "precision_at_k", "recall_at_k"]:
        lgb_v = results["lightgbm"][key]
        rf_v  = results["random_forest"][key]
        label = key.replace("_", " ").title()
        print(f"  {label:<23} {lgb_v:>12.4f} {rf_v:>14.4f}")
    print(f"  {'Train time (s)':<23} {lgb_train_sec:>12.1f} {rf_train_sec:>14.1f}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Save artifact
    # ------------------------------------------------------------------
    artifact = {
        "lgb_model":      lgb_clf,
        "rf_model":       rf_clf,
        "feature_names":  feature_names,
        "split_step":     split_step,
        "alert_budget_fraction": alert_budget_fraction,
        "metrics":        results,
        "model_version":  f"paysim-v1-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}",
        "provenance": {
            "git_sha":          _git_sha(),
            "data_path":        data_path,
            "data_sha256":      _sha256(data_path),
            "package_versions": _pkg_versions(),
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\n[PaySim] Artifact saved to {output_path}")

    # Save benchmark JSON
    benchmark = {
        "trained_at": artifact["model_version"],
        "split_step": split_step,
        "train_rows": len(train_df),
        "test_rows":  len(test_df),
        "alert_k":    alert_k,
        "results":    results,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2)
    print(f"[PaySim] Metrics saved to {metrics_path}")

    return artifact


if __name__ == "__main__":
    train_paysim_models()

